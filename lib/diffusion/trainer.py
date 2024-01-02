import os
import sys
import numpy as np

import logging
# Keep the import below for registering all model definitions
from .models import ddpm_res64, ddpm_res128

from . import losses
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
from . import sde_lib
import torch
from torch.utils import tensorboard
import sys
sys.path.append('/home/jialuo/CG_project/MeshDiffusion')
from .utils import save_checkpoint, restore_checkpoint
from ..dataset.shapenet_dmtet_dataset import ShapeNetDMTetDataset
from ..dataset.shapenet_dmtet_dataset_text import ShapeNetDMTetTextDataset
from ..dataset.shapenet_voxel_dataset import ShapeNetVoxelDataset
from ..dataset.shapenet_sdf_dataset import ShapeNetSdfDataset
from nvdiffrec.sd_utils import StableDiffusion

import torch
import sys
sys.path.append("/home/jialuo/CG_project")
from MeshDiffusion.lib.diffusion.models.ddpm_res64 import DDPMRes64Encode


def train(config):
    """Runs the training pipeline.

    Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """

    workdir = config.training.train_dir
    # Create directories for experimental logs
    logging.info("working dir: {:s}".format(workdir))


    tb_dir = os.path.join(workdir, "tensorboard")
    writer = tensorboard.SummaryWriter(tb_dir)

    resolution = config.data.image_size
    # Initialize model.
    score_model = mutils.create_model(config)
    if resolution == 128:
        score_model.to(dtype=torch.float16)
    total_params = sum(p.numel() for p in score_model.parameters())
    
    if config.training.classifier:
        print('Training Classifier!!!!!!!!!!')
        model_config = {
                "nf": 128,
                "ch_mult": (1, 1, 2, 4, 4),
                "num_res_blocks": 3,
                "attn_resolutions": (16,),
                "resamp_with_conv": True,
                "conditional": True,
                "dropout": 0.1,
                "name": "ddpm_res64_encode",
                "num_scale": 1000,
                "ema_rate": 0.9999,
                "normalization": "GroupNorm",
                "nonlinearity": "swish",
                "num_res_blocks_first": 2,
                "image_size": 64,
                "data": {
                    "centered": True,
                    "num_channels": 4
                },
        }
        
        unet_model = DDPMRes64Encode(**model_config).to(config.device)
        
    
    print(f"Total number of parameters: {total_params}")
    for n, p, in score_model.named_parameters():
        print(f"{n}: {p.numel()}")
    print(torch.cuda.memory_allocated(device=config.device))
    
    if config.training.classifier:
        optimizer = losses.get_optimizer(config, list(unet_model.parameters()))
        ema = ExponentialMovingAverage(unet_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=unet_model, ema=ema, step=0)
    else:
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)


    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])
    
    sd = None
    if config.training.twod_guide:
        sd = StableDiffusion(device=config.device)
        sd.eval()
        
    json_path = config.data.meta_path
    print("----- Assigning mask -----")
    logging.info(f"{json_path}, {config.data.filter_meta_path}")

    ### mask on tet to ignore regions
    if config.mask:
        mask = torch.load(f'./data/grid_mask_{resolution}.pt').view(1, 1, resolution, resolution, resolution).to("cuda")
    else:
        mask = torch.ones(1, 1, resolution, resolution, resolution).to("cuda", dtype=torch.float32)

    if hasattr(score_model.module, 'mask'):
        print("----- Assigning mask -----")
    score_model.module.mask.data[:] = mask[:]

    print(f"work dir: {workdir}")

    print("sdf normalized or not: ", config.data.normalize_sdf)
    if config.training.text_context:
        train_dataset = ShapeNetDMTetTextDataset(json_path, deform_scale=config.model.deform_scale, aug=True, grid_mask=mask, 
                filter_meta_path=config.data.filter_meta_path, normalize_sdf=config.data.normalize_sdf, extension='npy')
    elif config.data.dataset == "ShapeNet_voxel":
        train_dataset = ShapeNetVoxelDataset(json_path, deform_scale=config.model.deform_scale, aug=True, grid_mask=mask, 
                filter_meta_path=config.data.filter_meta_path, normalize_sdf=config.data.normalize_sdf, extension='binvox')
    elif config.data.dataset == "ShapeNet_sdf":
        train_dataset = ShapeNetSdfDataset(json_path, deform_scale=config.model.deform_scale, aug=True, grid_mask=mask, 
                filter_meta_path=config.data.filter_meta_path, normalize_sdf=config.data.normalize_sdf, extension='obj')    
    else:
        train_dataset = ShapeNetDMTetDataset(json_path, deform_scale=config.model.deform_scale, aug=True, grid_mask=mask, 
                filter_meta_path=config.data.filter_meta_path, normalize_sdf=config.data.normalize_sdf, extension='npy')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training.batch_size, 
            shuffle=True,
            num_workers=config.data.num_workers,
            pin_memory=True)

    data_iter = iter(train_loader)

    print("data loader set")
        
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)

    if not config.training.classifier:
        # Setup SDEs

        # Build one-step training and evaluation functions
        optimize_fn = losses.optimization_manager(config)
        train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                            mask=mask, loss_type=config.training.loss_type)
    else:
        optimize_fn = losses.optimization_manager(config)
        train_step_fn = losses.classifier_get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                            mask=mask, loss_type=config.training.loss_type)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step // config.training.iter_size,))

    iter_size = config.training.iter_size
    for step in range(initial_step // iter_size, num_train_steps + 1):
        tmp_loss = 0.0
        for step_inner in range(iter_size):
            try:
                # batch, batch_mask = next(data_iter)
                batch = next(data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader 
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = batch.cuda()

            # Execute one training step
            clear_grad_flag = (step_inner == 0)
            update_param_flag = (step_inner == iter_size - 1)
            if config.training.classifier:
                loss_dict = train_step_fn(state, batch, clear_grad=clear_grad_flag, update_param=update_param_flag, twod_guide=config.training.twod_guide, use_text_context=config.training.text_context, sd_model=None)
            else:
                loss_dict = train_step_fn(state, batch, clear_grad=clear_grad_flag, update_param=update_param_flag, twod_guide=config.training.twod_guide, use_text_context=config.training.text_context, sd_model=None, class_conditioned=config.model.use_spatial)
            loss = loss_dict['loss']
            tmp_loss += loss.item()

        tmp_loss /= iter_size
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, tmp_loss))
            sys.stdout.flush()
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            logging.info(f"save meta at iter {step}")
            save_checkpoint(checkpoint_meta_dir, state)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            logging.info(f"save model: {step}-th")
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'), state)
