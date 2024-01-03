# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch

from ..render import mesh
from ..render import render
from ..render import regularizer

import kaolin
from ..render import util as render_utils
import torch.nn.functional as F

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4, get_tet_gidx=False):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1,3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1,3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)

        if get_tet_gidx:
            face_to_valid_tet = torch.cat((
                tet_gidx[num_triangles == 1],
                torch.stack((tet_gidx[num_triangles == 2], tet_gidx[num_triangles == 2]), dim=-1).view(-1)
            ), dim=0)

            return verts, faces, uvs, uv_idx, face_to_valid_tet.long()
        else:
            return verts, faces, uvs, uv_idx

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometryFixedTopo(torch.nn.Module):
    def __init__(self, dmt_geometry, base_mesh, grid_res, scale, FLAGS, deform_scale=1.0, path=None, load_tet=False, **kwargs):
        super(DMTetGeometryFixedTopo, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()
        self.initial_guess = base_mesh
        self.scale         = scale
        self.tanh          = False
        self.deform_scale  = deform_scale
        self.path          = path
        self.load_tet      = load_tet
        tets = np.load('./data/tets/{}_tets_cropped.npz'.format(self.grid_res))

        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()
        if load_tet:
            tet = torch.load(str(path), map_location="cpu")
            self.sdf_sign = torch.nn.Parameter(tet['sdf'].to(FLAGS.device), requires_grad=False)
            self.deform = torch.nn.Parameter(tet['deform_unmasked'].to(FLAGS.device), requires_grad=True)
        else:    
            self.sdf_sign = torch.nn.Parameter(torch.sign(dmt_geometry.sdf.data + 1e-8).float(), requires_grad=False)
            self.sdf_sign.data[self.sdf_sign.data == 0] = 1.0 ## Avoid abiguity
            self.deform = torch.nn.Parameter(dmt_geometry.deform.data, requires_grad=True)
            
        self.register_parameter('sdf_sign', self.sdf_sign)

        self.sdf_abs    = torch.nn.Parameter(torch.ones_like(dmt_geometry.sdf), requires_grad=False)
        self.register_parameter('sdf_abs', self.sdf_abs)

        self.register_parameter('deform', self.deform)

        self.sdf_abs_ema = torch.nn.Parameter(self.sdf_abs.clone().detach(), requires_grad=False)
        self.deform_ema  = torch.nn.Parameter(self.deform.clone().detach(), requires_grad=False)

    def set_init_v_pos(self):
        with torch.no_grad():
            v_deformed = self.get_deformed()
            verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices)
            self.initial_guess_v_pos = verts


    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values

    def getVertNNDist(self):
        raise NotImplementedError
        v_deformed = (self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)).unsqueeze(0)
        return (pytorch3d.ops.knn.knn_points(v_deformed, v_deformed, K=2).dists[0, :, -1].detach()) ## K=2 because dist(self, self)=0

    def getMesh(self, material):
        # Run DM tet to get a base mesh
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx = self.marching_tets(v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def getMesh_tet_gidx(self, material):
        # Run DM tet to get a base mesh
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx = self.marching_tets(
            v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices, get_tet_gidx=True)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh, tet_gidx

        
    def update_ema(self, ema_coeff=0.9):
        return

    def get_deformed(self):
        if self.tanh:
            v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform) * self.deform_scale
        else:
            v_deformed = self.verts + 2 / (self.grid_res * 2) * self.deform * self.deform_scale
        return v_deformed

    def getValidTetIdx(self):
        # Run DM tet to get a base mesh
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx = self.marching_tets(
            v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices, get_tet_gidx=True)
        return tet_gidx.long()

    def getValidVertsIdx(self):
        # Run DM tet to get a base mesh
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx = self.marching_tets(
            v_deformed, self.sdf_sign * self.sdf_abs.abs(), self.indices, get_tet_gidx=True)
        return self.indices[tet_gidx.long()].unique()

    def getTetCenters(self):
        v_deformed = self.get_deformed() # size: N x 3
        face_verts = v_deformed[self.indices] # size: M x 4 x 3
        face_centers = face_verts.mean(dim=1) # size: M x 3

        return face_centers

    def clamp_deform(self):
        if not self.tanh:
            self.deform.data[:] = self.deform.data.clamp(-0.99, 0.99)

    def render(self, glctx, target, lgt, opt_material, bsdf=None, ema=False, xfm_lgt=None, get_visible_tets=False):
        opt_mesh = self.getMesh(opt_material)
        tet_centers = self.getTetCenters() if get_visible_tets else None
        return render.render_mesh(
            glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt, tet_centers=tet_centers, FLAGS=self.FLAGS)


    def render_with_mesh(self, glctx, target, lgt, opt_material, bsdf=None, xfm_lgt=None):
        opt_mesh = self.getMesh(opt_material)
        return opt_mesh, render.render_mesh(
            glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            num_layers=self.FLAGS.layers, msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt, FLAGS=self.FLAGS)
        
    def gen_target(self, batch):
        iter_res = self.FLAGS.train_res
        proj_mtx = render_utils.perspective(np.deg2rad(45), iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        ang    = np.random.rand() * np.pi * 2
        mv     = render_utils.translate(0, 0, -2.0) @ (render_utils.rotate_x(-0.4) @ render_utils.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        target = {}
        target['mv'] = mv[None, ...].cuda()
        target['envlight_transform'] = None
        target['mvp'] = mvp[None, ...].cuda()
        target['campos'] = campos[None, ...].cuda()
        target['resolution'] = self.FLAGS.display_res
        target['spp'] = self.FLAGS.spp
        target['background'] = torch.ones([batch, 1000, 1000, 3], dtype=torch.float32, device='cuda')
        
        # print(target['mvp'])
        # print(mvp.shape)
        return ang, target
        
    def gen_img(self, glctx, lgt, opt_material, xfm_lgt=None, val_save_name=None, batch=None):
        target_list = []
        ang_list = []
        for i in range(batch):
            ang, target = self.gen_target(batch)
            target_list.append(target)
            ang = ang * 180 / np.pi - 180
            ang_list.append(torch.tensor(ang, dtype=torch.float32, device='cuda'))
        final_target = {
                'mv' : torch.cat(list([item['mv'] for item in target_list]), dim=0),
                'mvp' : torch.cat(list([item['mvp'] for item in target_list]), dim=0),
                'campos' : torch.cat(list([item['campos'] for item in target_list]), dim=0),
                'resolution' : self.FLAGS.display_res,
                'spp' : self.FLAGS.spp,
                'background': torch.ones([batch, 1000, 1000, 3], dtype=torch.float32, device='cuda')
            }
        # imesh, buffers = self.render_with_mesh(glctx, final_target, lgt, opt_material, xfm_lgt=xfm_lgt)
        buffers = self.render(glctx, final_target, lgt, opt_material, ema=True)
        img_list = []
        for i in range(batch):
            img = render_utils.rgb_to_srgb(buffers['shaded'][...,0:3])[i]
            np_img = img.detach().cpu().numpy()
            render_utils.save_image(f'{val_save_name}_{i}.png', np_img)
            img_list.append(img)    
        img_list = torch.stack(img_list, dim=0)
        ang_list = torch.stack(ang_list, dim=0)
        return ang_list, img_list   

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, with_reg=True, xfm_lgt=None, no_depth_thin=True):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        imesh, buffers = self.render_with_mesh(glctx, target, lgt, opt_material, xfm_lgt=xfm_lgt)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) 
        img_loss = img_loss + loss_fn(
            buffers['shaded'][..., 0:3] * color_ref[..., 3:], 
            color_ref[..., 0:3] * color_ref[..., 3:]
        )
        mask = target['mask'][:, :, :, 0]


        if no_depth_thin:
            valid_depth_mask = (
                (target['depth_second'] >= 0).float() * ((target['depth_second'] - target['depth']).abs() >= 5e-3).float()
            ).detach()
        else:
            valid_depth_mask = 1.0

        depth_diff = (buffers['depth'][:, :, :, :1] - target['depth'][:, :, :, :1]).abs() * mask.unsqueeze(-1) * valid_depth_mask
        depth_diff = (buffers['depth_second'][:, :, :, :1] - target['depth_second'][:, :, :, :1]).abs() * mask.unsqueeze(-1) * valid_depth_mask * 1e-1

        l1_loss_mask = (depth_diff < 1.0).float()
        img_loss = img_loss + (l1_loss_mask * depth_diff + (1 - l1_loss_mask) * depth_diff.pow(2)).mean() * 100.0

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer. 
        reg_loss += regularizer.laplace_regularizer_const(imesh.v_pos - self.initial_guess_v_pos, imesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter) * 1e-2
        
        ### Chamfer distance for ShapeNet
        pred_points = kaolin.ops.mesh.sample_points(imesh.v_pos.unsqueeze(0), imesh.t_pos_idx, 50000)[0][0]
        target_pts = target['spts']
        chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), target_pts.unsqueeze(0)).mean()
        reg_loss += chamfer

        return img_loss, reg_loss
    
    def tick_text(self, glctx, lgt, opt_material, prompt=None, sd=None, FLAGS=None, iteration=None):

        self.deform.requires_grad = True
        
            
        # if FLAGS.normal_only:
        #         base_mesh = xatlas_uvmap_nrm(glctx, geometry, mat, FLAGS)
        #     else:
        #         base_mesh = xatlas_uvmap(glctx, geometry, mat, FLAGS)
    
        # geometry = DMTetGeometryFixedTopo(
        #         geometry, base_mesh, FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS, 
        #         deform_scale=FLAGS.second_stage_deform
        #     )
        print("---------------------------------------------------")
        print(self.getAABB())
        ang_list, img_list = self.gen_img(glctx, lgt, opt_material, val_save_name=FLAGS.val_save_name, batch=FLAGS.batch)
        # img_list: [B, H, W, 3]
        img_list = img_list.permute(0, 3, 1, 2)
        text_embs = sd.get_text_all_embeddings(prompt, '', ang_list)
        
        img_loss = sd.train_step(text_embs, img_list)
        
        # if iteration > 200 and iteration < 2000 and iteration % 20 == 0:
        #     with torch.no_grad():
        #         v_pos = self.get_deformed()
        #         v_pos_camera_homo = ru.xfm_points(v_pos[None, ...], target['mvp'])
        #         v_pos_camera = v_pos_camera_homo[:, :, :2] / v_pos_camera_homo[:, :, -1:]
        #         v_pos_camera_discrete = torch.round((v_pos_camera * 0.5 + 0.5).clip(0, 1) * (target['resolution'][0] - 1)).long()
        #         mask_cont = F.conv2d(target['mask_cont'][:, :, :, 0].unsqueeze(1), self.smooth_kernel, stride=1, padding=self.padding)[:, 0]
        #         target_mask = mask_cont == 0
        #         for k in range(target_mask.size(0)):
        #             assert v_pos_camera_discrete[k].min() >= 0 and v_pos_camera_discrete[k].max() < target['resolution'][0]
        #             v_mask = target_mask[k, v_pos_camera_discrete[k, :, 1], v_pos_camera_discrete[k, :, 0]].view(v_pos.size(0))
        #             self.sdf.data[v_mask] = 1e-2
        #             self.deform.data[v_mask] = 0.0

        # # ==============================================================================================
        # #  Render optimizable object with identical conditions
        # # ==============================================================================================
        # imesh, buffers = self.render_with_mesh(glctx, target, lgt, opt_material, noise=0.0, xfm_lgt=xfm_lgt)
        imesh = self.getMesh(opt_material)

        # # ==============================================================================================
        # #  Compute loss
        # # ==============================================================================================
        # t_iter = iteration / self.FLAGS.iter

        # # Image-space loss, split into a coverage component and a color component
        # color_ref = target['img']
        # img_loss = torch.tensor(0.0).cuda()
        # alpha_scale = 1.0
        # img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:]) * alpha_scale
        # img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])


        # color_ref_second = target['img_second']
        # img_loss = img_loss + torch.nn.functional.mse_loss(buffers['shaded_second'][..., 3:], color_ref_second[..., 3:]) * alpha_scale * 1e-1
        # img_loss = img_loss + loss_fn(buffers['shaded_second'][..., 0:3] * color_ref_second[..., 3:], color_ref_second[..., 0:3] * color_ref_second[..., 3:]) * 1e-1

        # mask = (target['mask_cont'][:, :, :, 0] == 1.0).float()

        # if iteration < 10000:
        #     depth_scale = 100.0
        # else:
        #     depth_scale = 1.0

        # if iteration % 300 == 0 and iteration < 1790:
        #     self.deform.data[:] *= 0.4

        # if no_depth_thin:
        #     valid_depth_mask = (target['depth_second'] >= 0).float().detach()
        #     depth_prox_mask = ((target['depth_second'] - target['depth']).abs() >= 5e-3).float().detach()
        # else:
        #     valid_depth_mask = 1.0

        # depth_diff = (buffers['depth'][:, :, :, :1] - target['depth'][:, :, :, :1]).abs() * mask.unsqueeze(-1) * valid_depth_mask
        # depth_diff_second = (buffers['depth_second'][:, :, :, :1] - target['depth_second'][:, :, :, :1]).abs() * mask.unsqueeze(-1) * valid_depth_mask * depth_prox_mask * 1e-1

        # thres = 1.0
        # l1_loss_mask = (depth_diff < thres).float()
        # l1_loss_mask_second = (depth_diff_second < thres).float()
        
        # img_loss = img_loss + (
        #     (
        #         l1_loss_mask * depth_diff 
        #         + (1 - l1_loss_mask) * (depth_diff.pow(2) + thres - thres**2)
        #     ).mean() * 1.0 * depth_scale
        #     + (
        #         l1_loss_mask_second * depth_diff_second 
        #         + (1 - l1_loss_mask_second) * (depth_diff_second.pow(2) + thres - thres**2)
        #     ).mean() * 1.0 * depth_scale
        # )


        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer. 
        t_iter = iteration / self.FLAGS.iter
        reg_loss += regularizer.laplace_regularizer_const(imesh.v_pos - self.initial_guess_v_pos, imesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter) * 1e-2
        
        ### Chamfer distance for ShapeNet
        # pred_points = kaolin.ops.mesh.sample_points(imesh.v_pos.unsqueeze(0), imesh.t_pos_idx, 50000)[0][0]
        # target_pts = target['spts']
        # chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), target_pts.unsqueeze(0)).mean()
        # reg_loss += chamfer

        # Albedo (k_d) smoothnesss regularizer
        # reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        # reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 1e0 * min(1.0, iteration / 500)

        # pointcloud chamfer distance
        # pred_points = kaolin.ops.mesh.sample_points(imesh.v_pos.unsqueeze(0), imesh.t_pos_idx, 50000)[0][0]
        # target_pts = target['spts']
        # chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), target_pts.unsqueeze(0)).mean()

        # reg_loss += chamfer


        return img_loss, reg_loss
    
    def tick_voxel(self, glctx, target, lgt, opt_material, loss_fn, iteration, with_reg=True, xfm_lgt=None, no_depth_thin=True):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        imesh = self.getMesh(opt_material)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================
        t_iter = iteration / self.FLAGS.iter
        valid_depth_mask = 1.0

        reg_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # Compute regularizer. 
        reg_loss += regularizer.laplace_regularizer_const(imesh.v_pos - self.initial_guess_v_pos, imesh.t_pos_idx) * self.FLAGS.laplace_scale * (1 - t_iter) * 1e-2
        
        ### Chamfer distance for ShapeNet
        pred_points = kaolin.ops.mesh.sample_points(imesh.v_pos.unsqueeze(0), imesh.t_pos_idx, 50000)[0][0]
        target_pts = target['spts']
        chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), target_pts.unsqueeze(0)).mean()
        reg_loss += chamfer

        return torch.tensor(0.0).cuda(), reg_loss