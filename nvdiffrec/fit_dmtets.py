# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import time
import argparse
import json
import sys

import cv2
import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

# Import data readers / generators
from lib.dataset.dataset_mesh import DatasetMesh
from lib.dataset.dataset_shapenet import ShapeNetDataset

# Import topology / geometry trainers
from lib.geometry.dmtet import DMTetGeometry
from lib.geometry.dmtet_fixedtopo import DMTetGeometryFixedTopo

import lib.render.renderutils as ru
from lib.render import obj
from lib.render import material
from lib.render import util
from lib.render import mesh
from lib.render import texture
from lib.render import mlptexture
from lib.render import light
from lib.render import render
from torchviz import make_dot

import sys
sys.path.append('/home/jialuo/CG_project/MeshDiffusion')

from nvdiffrec.sd_utils import StableDiffusion
import traceback
from imageio import imread

RADIUS = 2.0

# # Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

# define colors
color1 = (0, 0, 255)     #red
color2 = (0, 165, 255)   #orange
color3 = (0, 255, 255)   #yellow
color4 = (255, 255, 0)   #cyan
color5 = (255, 0, 0)     #blue
color6 = (128, 64, 64)   #violet
colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)

# resize lut to 256 (or more) values
lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background
    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    target['spts'] = target['spts'].cuda()
    target['vpts'] = target['vpts'].cuda()
    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

@torch.no_grad()
def xatlas_uvmap_nrm(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, normal = render.render_uv_nrm(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : mat['kd'],
        'ks'     : mat['ks'],
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat

def initial_guess_material_knownkskd(geometry, mlp, FLAGS, init_mat=None):
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    if mlp:
        mlp_min = nrm_min
        mlp_max = nrm_max
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=3, min_max=[mlp_min, mlp_max])
        # mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=3, min_max=None)
        mat =  material.Material({
            'kd'     : init_mat['kd'],
            'ks'     : init_mat['ks'],
            'normal' : mlp_map_opt,
        })
    else:
        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : init_mat['kd'],
            'ks'     : init_mat['ks'],
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat

###############################################################################
# Validation & testing
###############################################################################

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])
        lgt.xfm(target['envlight_transform'])

        try:
            buffers = geometry.render(glctx, target, lgt, opt_material, ema=True, xfm_lgt=target['envlight_transform'])
        except:
            buffers = geometry.render(glctx, target, lgt, opt_material, xfm_lgt=target['envlight_transform'])

        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

        return result_image, result_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        print("Running validation")
        for it, target in enumerate(dataloader_validate):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)
            # print('-------------------------------------------------------------------------')
            # print(target)
            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
           
            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))

            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    return avg_psnr

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []
        try:
            self.sdf_params = [self.geometry.sdf]
        except:
            self.sdf_params = []
        self.deform_params = [self.geometry.deform]

    def forward(self, target, it, prompt=None, sd=None):
        # print(f"AAAAAAAAAAAAAAAAAAAA")
        if target is not None:
            if self.optimize_light:
                self.light.build_mips()
                if FLAGS.voxel:
                    return self.geometry.tick_voxel(glctx, target, self.light, self.material, self.image_loss_fn, it, xfm_lgt=None)      
                if self.FLAGS.camera_space_light:
                    self.light.xfm(target['mv'])
            if FLAGS.voxel:
                return self.geometry.tick_voxel(glctx, target, self.light, self.material, self.image_loss_fn, it, xfm_lgt=None)      
            self.light.xfm(target['envlight_transform'])
            return self.geometry.tick(glctx, target, self.light, self.material, self.image_loss_fn, it, xfm_lgt=target['envlight_transform'])
        else:
            # print(f"BBBBBBBBBBBBBBB")
            return self.geometry.tick_text(glctx, self.light, self.material, prompt=prompt, sd=sd, FLAGS=self.FLAGS, iteration=it)
def optimize_mesh(
        glctx,
        geometry,
        opt_material,
        lgt,
        dataset_train,
        dataset_validate,
        FLAGS,
        warmup_iter=0,
        log_interval=10,
        pass_idx=0,
        pass_name="",
        optimize_light=True,
        optimize_geometry=True,
        prompt = None,
    ):

    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================

    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter 
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002)) # Exponential falloff from [1.0, 0.1] over 5k epochs.    

    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, FLAGS)

    if FLAGS.multi_gpu: 
        # raise NotImplementedError
        # Multi GPU training mode
        import apex
        from apex.parallel import DistributedDataParallel as DDP

        trainer = DDP(trainer_noddp)
        trainer.train()
        if optimize_geometry:
            optimizer_mesh = apex.optimizers.FusedAdam(trainer_noddp.geo_params, lr=learning_rate_pos)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

        optimizer = apex.optimizers.FusedAdam(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 
    else:
        # Single GPU training mode
        trainer = trainer_noddp
        if optimize_geometry:
            # optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params, lr=learning_rate_pos)
            optimizer_mesh = torch.optim.Adam([
                {'params': trainer_noddp.sdf_params, 'lr': learning_rate_pos},
                {'params': trainer_noddp.deform_params, 'lr': learning_rate_pos},
            ])
            # optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params, lr=learning_rate_pos, betas=(0.2, 0.999), eps=1e-5)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

        optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
        # optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat, betas=(0.2, 0.999), eps=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9)) 

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    if dataset_train is not None:
        dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    else:
        dataloader_train = None

    print("Start training loop...")
    sys.stdout.flush()
    
    if dataset_train is None and not FLAGS.voxel:
        sd = StableDiffusion('cuda')
    else:
        sd = None
        
    if FLAGS.voxel:
        data = torch.from_numpy(np.load(FLAGS.voxel_path)).to('cuda', dtype=torch.float32)
        target = {
            'spts': data
        }

    if dataset_train is None or FLAGS.voxel:
        for it in range(0, FLAGS.iter):

            # # Mix randomized background into dataset image
            # target = prepare_batch(target, 'random')
            
            if not FLAGS.voxel:
                target = None

            iter_start_time = time.time()


            # ==============================================================================================
            #  Zero gradients
            # ==============================================================================================
            optimizer.zero_grad()
            if optimize_geometry:
                optimizer_mesh.zero_grad()

            # ==============================================================================================
            #  Training
            # ==============================================================================================
            img_loss, reg_loss = trainer(target, it, prompt, sd)

            # ==============================================================================================
            #  Final loss
            # ==============================================================================================
            total_loss = img_loss + reg_loss

            img_loss_vec.append(img_loss.item())
            reg_loss_vec.append(reg_loss.item())

            # ==============================================================================================
            #  Backpropagate
            # ==============================================================================================
            # graph = make_dot(total_loss)
            # graph.view('model_structure.pdf', './')
            total_loss.backward()

            if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
                lgt.base.grad *= 64
            if 'kd_ks_normal' in opt_material:
                opt_material['kd_ks_normal'].encoder.params.grad /= 8.0
            if 'normal' in opt_material and FLAGS.normal_only:
                try:
                    opt_material['normal'].encoder.params.grad /= 8.0
                except:
                    pass

            optimizer.step()
            scheduler.step()

            if optimize_geometry:
                optimizer_mesh.step()
                scheduler_mesh.step()

            geometry.clamp_deform()
            geometry.update_ema()

            # ==============================================================================================
            #  Clamp trainables to reasonable range
            # ==============================================================================================
            with torch.no_grad():
                if 'kd' in opt_material:
                    opt_material['kd'].clamp_()
                if 'ks' in opt_material:
                    opt_material['ks'].clamp_()
                if 'normal' in opt_material and not FLAGS.normal_only:
                    opt_material['normal'].clamp_()
                    opt_material['normal'].normalize_()
                if lgt is not None:
                    lgt.clamp_(min=0.0)

            torch.cuda.current_stream().synchronize()
            iter_dur_vec.append(time.time() - iter_start_time)

            # ==============================================================================================
            #  Logging
            # ==============================================================================================
            if it % log_interval == 0 and FLAGS.local_rank == 0:
                img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
                reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
                iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
                
                remaining_time = (FLAGS.iter-it)*iter_dur_avg
                print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                    (it, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
                sys.stdout.flush()
        
        # release sd
        if not FLAGS.voxel:
            del sd
        torch.cuda.empty_cache()
    else:
        for it, target in enumerate(dataloader_train):

            # Mix randomized background into dataset image
            print(target['img'].shape)
            print('~'*100)
            target = prepare_batch(target, 'random')
            # print('-------------------------------------------------------------------------')
            # print(target)
            iter_start_time = time.time()


            # ==============================================================================================
            #  Zero gradients
            # ==============================================================================================
            optimizer.zero_grad()
            if optimize_geometry:
                optimizer_mesh.zero_grad()

            # ==============================================================================================
            #  Training
            # ==============================================================================================
            img_loss, reg_loss = trainer(target, it)

            # ==============================================================================================
            #  Final loss
            # ==============================================================================================
            total_loss = img_loss + reg_loss

            img_loss_vec.append(img_loss.item())
            reg_loss_vec.append(reg_loss.item())

            # ==============================================================================================
            #  Backpropagate
            # ==============================================================================================
            total_loss.backward()

            if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:
                lgt.base.grad *= 64
            if 'kd_ks_normal' in opt_material:
                opt_material['kd_ks_normal'].encoder.params.grad /= 8.0
            if 'normal' in opt_material and FLAGS.normal_only:
                try:
                    opt_material['normal'].encoder.params.grad /= 8.0
                except:
                    pass

            optimizer.step()
            scheduler.step()

            if optimize_geometry:
                optimizer_mesh.step()
                scheduler_mesh.step()

            geometry.clamp_deform()
            geometry.update_ema()

            # ==============================================================================================
            #  Clamp trainables to reasonable range
            # ==============================================================================================
            with torch.no_grad():
                if 'kd' in opt_material:
                    opt_material['kd'].clamp_()
                if 'ks' in opt_material:
                    opt_material['ks'].clamp_()
                if 'normal' in opt_material and not FLAGS.normal_only:
                    opt_material['normal'].clamp_()
                    opt_material['normal'].normalize_()
                if lgt is not None:
                    lgt.clamp_(min=0.0)

            torch.cuda.current_stream().synchronize()
            iter_dur_vec.append(time.time() - iter_start_time)

            # ==============================================================================================
            #  Logging
            # ==============================================================================================
            if it % log_interval == 0 and FLAGS.local_rank == 0:
                img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))
                reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
                iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))
                
                remaining_time = (FLAGS.iter-it)*iter_dur_avg
                print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % 
                    (it, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
                sys.stdout.flush()

    return geometry, opt_material

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    # sleep(randint(0,15))

    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default='./configs/res64.json', help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=3000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default='./dmtet_results_test_16')
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('-ind', '--index', type=int)
    parser.add_argument('-ss', '--split-size', type=int, default=10)
    parser.add_argument('--cropped', type=bool, default=True)
    parser.add_argument('-no', '--normal-only', type=bool, default=True)
    parser.add_argument('--meta-path', type=str, default='./data/shapenet_json/cat.json')
    parser.add_argument('-rp', '--resume-path', type=str, default=None)
    parser.add_argument('-ema', '--use-ema', action="store_true")
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--val_save_name', type=str, default='val')
    parser.add_argument('--load_tet_path', type=str, default=None)
    parser.add_argument('--load_tet', action='store_true', default=False)
    parser.add_argument('--voxel', action='store_true', default=False)
    parser.add_argument('--voxel_path', type=str, default=None)
    FLAGS = parser.parse_args()
    print(f"parsed arguments")
    global_index = FLAGS.index * FLAGS.split_size
    print("------------------ssss------------------------------------------")
    print(f"FLAGS.normal_only {FLAGS.normal_only}")
    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 1.0                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = True                   # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for sdf regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = False
    FLAGS.cropped             = True
    FLAGS.use_ema             = False
    FLAGS.random_lgt          = True
    FLAGS.dataset_flat_shading = False

    FLAGS.local_rank = 0
    torch.cuda.set_device(FLAGS.device)
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    print("------------------ssss------------------------------------------")
    print(f"FLAGS.normal_only {FLAGS.normal_only}")
    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")
        
    if FLAGS.prompt is not None:
        FLAGS.out_dir = os.path.join(FLAGS.out_dir, 'text_prompt')

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.out_dir, 'val_viz'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.out_dir, 'tets'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.out_dir, 'tets_pre'), exist_ok=True)
    
    print(f"batch size {FLAGS.batch}")


    print(f"Using dmt grid of resolution {FLAGS.dmtet_grid}")

    glctx = dr.RasterizeCudaContext()    

    img = imread('/home/jialuo/CG_project/MeshDiffusion/nvdiffrec/img_tinted.png')
    x_img = torch.tensor(img, dtype=torch.float32, device='cuda') / 255.0
    ### Default mtl
    mtl_default = {
        'name' : '_default_mat',
        'bsdf': 'diffuse',
        'uniform': True,
        # 'kd'   : texture.Texture2D(torch.tensor([0.75, 0.3, 0.6], dtype=torch.float32, device='cuda'), trainable=False),
        # 'kd'   : texture.Texture2D(torch.tensor([0.75, 0.3, 0.6], dtype=torch.float32, device='cuda')[None, None, ...].repeat(64, 64, 1), trainable=True),
        # 'kd'   : texture.Texture2D(torch.tensor([0.1, 0.9, 0.1], dtype=torch.float32, device='cuda'), trainable=True),
        # 'kd'   : texture.Texture2D(torch.randn((512, 512, 3), device='cuda', dtype=torch.float32), trainable=True),
        # 'kd'   : texture.Texture2D(torch.full((512, 512, 3), 1.0, device='cuda', dtype=torch.float32), trainable=True),
        'kd'   : texture.Texture2D(x_img, trainable=(FLAGS.prompt is not None)),
        'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'), trainable=False)
    }


    print(f"meta json path {FLAGS.meta_path}")
    if FLAGS.prompt is None:
        shapenet_dataset = ShapeNetDataset(f'{FLAGS.meta_path}')
    else:
        shapenet_dataset = None

    print("Start iterating through objects")
    sys.stdout.flush()

    for k in range(0, FLAGS.split_size):
        if (FLAGS.prompt is not None or FLAGS.voxel) and k != 0:
            break
        # ==============================================================================================
        #  Create data pipeline
        # ==============================================================================================

        global_index = k + FLAGS.index * FLAGS.split_size

        print("file path to save: {:s}".format(os.path.join(FLAGS.out_dir, 'tets/dmt_dict_{:05d}.pt'.format(global_index))))

        skip_if_exists = True
        if skip_if_exists and os.path.exists(os.path.join(FLAGS.out_dir, 'tets/dmt_dict_{:05d}.pt'.format(global_index))):
            continue

        try:
            if FLAGS.prompt is None and not FLAGS.voxel:
                global_index = k + FLAGS.index * FLAGS.split_size

                if global_index >= len(shapenet_dataset):
                    break
                mesh_fname = shapenet_dataset[global_index]

                print(f"Loading mesh: {mesh_fname}")
                sys.stdout.flush()
                ref_mesh = mesh.load_mesh(mesh_fname, FLAGS.mtl_override, mtl_default, use_default=FLAGS.normal_only, no_additional=True)
                ref_mesh = mesh.center_by_reference(ref_mesh, mesh.aabb_clean(ref_mesh), 1.0)

                a = ref_mesh.v_nrm.clone()
                ref_mesh = mesh.auto_normals(ref_mesh) ### important
                print("Loading dataset")
                sys.stdout.flush()
                dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=False)
                dataset_validate = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, validate=True)
                print("Dataset loaded")
                sys.stdout.flush()
            else:
                dataset_train = None
                dataset_validate = None


            # ==============================================================================================
            #  Create env light with trainable parameters
            # ==============================================================================================
            
            if FLAGS.learn_light:
                lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
            else:
                lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale, trainable=False)

            # ==============================================================================================
            #  If no initial guess, use DMtets to create geometry
            # ==============================================================================================

            # Setup geometry for optimization
            geometry = DMTetGeometry(
                FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS, 
                deform_scale=FLAGS.first_stage_deform, path=FLAGS.load_tet_path, load_tet=FLAGS.load_tet
            )

            # Setup textures, make initial guess from reference if possible
            if not FLAGS.normal_only:
                mat = initial_guess_material(geometry, True, FLAGS, mtl_default)
            else:
                mat = initial_guess_material_knownkskd(geometry, False, FLAGS, mtl_default)

            print("Start optimization")
            sys.stdout.flush()

            if FLAGS.resume_path is None:
                # Run optimization
                if FLAGS.prompt is None:
                    geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, 
                                FLAGS, pass_idx=0, pass_name="dmtet_pass1", optimize_light=FLAGS.learn_light, prompt=FLAGS.prompt, optimize_geometry=not FLAGS.lock_pos)

                base_mesh = geometry.getMesh(mat)
                vert_mask = torch.zeros_like(geometry.sdf).long().cuda().view(-1, 1)
                vert_mask[geometry.getValidVertsIdx()] = 1

                # Free temporaries / cached memory 
                torch.cuda.empty_cache() ### may slow down training

                torch.save({
                    'sdf': geometry.sdf.cpu().detach(),
                    'sdf_ema': geometry.sdf_ema.cpu().detach(),
                    'deform': (geometry.deform * vert_mask).cpu().detach(),
                    'deform_unmasked': geometry.deform.cpu().detach(),
                }, os.path.join(FLAGS.out_dir, 'tets_pre/dmt_dict_{:05d}.pt'.format(global_index)))

                old_geometry = geometry
            else:
                dmt_dict = torch.load(os.path.join(FLAGS.resume_path, 'tets_pre/dmt_dict_{:05d}.pt'.format(global_index)))
                if FLAGS.use_ema:
                    geometry.sdf.data[:] = dmt_dict['sdf_ema']
                else:
                    geometry.sdf.data[:] = dmt_dict['sdf']
                geometry.deform.data[:] = dmt_dict['deform']
                old_geometry = geometry
            
                # Create textured mesh from result
                if FLAGS.normal_only:
                    base_mesh = xatlas_uvmap_nrm(glctx, geometry, mat, FLAGS)
                else:
                    base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)


            # # ==============================================================================================
            # #  Pass 2: Finetune deformation with fixed topology
            # # ==============================================================================================
            geometry = DMTetGeometryFixedTopo(
                geometry, base_mesh, FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS, 
                deform_scale=FLAGS.second_stage_deform, path=FLAGS.load_tet_path, load_tet=FLAGS.load_tet
            )
            
            if FLAGS.prompt is None:
                geometry.sdf_sign.requires_grad = False
                geometry.sdf_abs.requires_grad = False
                geometry.deform.requires_grad = True

                geometry.deform.data[:] = geometry.deform * FLAGS.first_stage_deform / FLAGS.second_stage_deform

                if FLAGS.use_ema:
                    geometry.sdf_sign.data[:] = torch.sign(old_geometry.sdf_ema)
                else:
                    geometry.sdf_sign.data[:] = torch.sign(old_geometry.sdf)
            
            geometry.set_init_v_pos()


            geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, FLAGS, 
                        pass_idx=1, pass_name="mesh_pass", warmup_iter=100, optimize_light=FLAGS.learn_light and not FLAGS.lock_light, 
                        optimize_geometry=not FLAGS.lock_pos, prompt=FLAGS.prompt)

            vert_mask = torch.zeros_like(geometry.sdf_sign).long().cuda().view(-1, 1)
            vert_mask[geometry.getValidVertsIdx()] = 1

            torch.save({
                    'sdf': geometry.sdf_sign.cpu().detach(),
                    'deform': (geometry.deform * vert_mask).cpu().detach(),
                    'deform_unmasked': geometry.deform.cpu().detach(),
                }, 
                os.path.join(FLAGS.out_dir, 'tets/dmt_dict_{:05d}.pt'.format(global_index))
            )
            
            
            if FLAGS.local_rank == 0 and FLAGS.validate and FLAGS.prompt is None:
                validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, f"val_viz/dmtet_validate_{FLAGS.index}_{k}_{FLAGS.split_size}"), FLAGS)
            os.makedirs(os.path.join(FLAGS.out_dir, 'mtls'), exist_ok=True)
            mesh_ = geometry.getMesh(mat)
            obj.write_obj(os.path.join(FLAGS.out_dir, 'mtls'), mesh_)
            # Free temporaries / cached memory
            del geometry
            if FLAGS.prompt is None:
                del ref_mesh
                del dataset_train
                del dataset_validate
            torch.cuda.empty_cache() ### may slow down training

            print(f"\n\n============ {FLAGS.index}_{k}/{FLAGS.split_size} finished ============\n\n")
        except Exception as err:
            print(f"\n\n============ {FLAGS.index}_{k}/{FLAGS.split_size} Failed ============\n\n")
            print(traceback.format_exc())
            print("\n\n")
            continue
            


