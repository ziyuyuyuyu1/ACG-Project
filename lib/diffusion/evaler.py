import os
import sys
import numpy as np

import logging
from . import losses
from .models import utils as mutils
from .models.ema import ExponentialMovingAverage
from . import sde_lib
import torch
import torch.nn.functional as F
from .utils import restore_checkpoint
from . import sampling

sys.path.append("/home/jialuo/CG_project")
from MeshDiffusion.lib.diffusion.models.ddpm_res64 import DDPMRes64Encode

def uncond_gen(
        config,
        idx=0,
    ):
    """
        Unconditional Generation
    """
    with torch.no_grad():
        eval_dir, ckpt_path = config.eval.eval_dir, config.eval.ckpt_path
        # Create directory to eval_folder
        os.makedirs(eval_dir, exist_ok=True)

        scaler, inverse_scaler = lambda x: x, lambda x: x

        # Initialize model
        score_model = mutils.create_model(config)
        
        classifier_fn = None
        
        if config.training.classifier:
            print(f'using classifier sclae {config.eval.classifier_scale}')
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
            unet_model.load_state_dict(torch.load(config.eval.classifier_path)['model'])
            unet_model.eval()
            
            def classifier_fn(x, t, y=None):
                assert y is not None
                with torch.enable_grad():
                    y = y - 1
                    x_in = x.detach().requires_grad_(True)
                    logits = unet_model(x_in, t)
                    log_probs = F.log_softmax(logits, dim=-1)
                    selected = log_probs[range(len(logits)), y.view(-1)]
                    # print(y.view(-1))
                    # print(selected.shape)
                    return torch.autograd.grad(selected.sum(), x_in)[0] * config.eval.classifier_scale
        
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        # Setup SDEs
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
        print(f'SDE N {sde.N}')

        img_size = config.data.image_size
        if config.mask:
            grid_mask = torch.load(f'./data/grid_mask_{img_size}.pt').view(1, img_size, img_size, img_size).to("cuda")
        else:
            grid_mask = 1.


        sampling_eps = 1e-3
        sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        config.data.image_size, config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, grid_mask=grid_mask, classifier_fn=classifier_fn)

        assert os.path.exists(ckpt_path)
        print('ckpt path:', ckpt_path)
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            raise
        ema.copy_to(score_model.parameters())

        print(f"loaded model is trained till iter {state['step'] // config.training.iter_size}")
        save_file_path = os.path.join(eval_dir, f"{idx}.npy")


        samples, n = sampling_fn(score_model, context=torch.tensor([[1],[2],[3],[4],[5]]))
        samples = samples.cpu().numpy()
        np.save(save_file_path, samples)


def slerp(z1, z2, alpha):
    '''
        Spherical Linear Interpolation
    '''
    theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
    return (
            torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
            + torch.sin(alpha * theta) / torch.sin(theta) * z2
    )

def uncond_gen_interp(
        config,
        idx=0,
    ):
    """
        Generation with interpolation between initial noises
        Used for DDIM
    """
    with torch.no_grad():
        eval_dir, ckpt_path = config.eval.eval_dir, config.eval.ckpt_path
        # Create directory to eval_folder
        os.makedirs(eval_dir, exist_ok=True)

        scaler, inverse_scaler = lambda x: x, lambda x: x

        # Initialize model
        score_model = mutils.create_model(config)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        # Setup SDEs
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)

        img_size = config.data.image_size
        grid_mask = torch.load(f'./data/grid_mask_{img_size}.pt').view(1, img_size, img_size, img_size).to("cuda")

        sampling_eps = 1e-3
        sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        config.data.image_size, config.data.image_size, config.data.image_size)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, grid_mask=grid_mask)

        assert os.path.exists(ckpt_path)
        print('ckpt path:', ckpt_path)
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            raise
        ema.copy_to(score_model.parameters())

        print(f"loaded model is trained till iter {state['step'] // config.training.iter_size}")
        save_file_path = os.path.join(eval_dir, f"{idx}.npy")


        noise = sde.prior_sampling(
            (2, config.data.num_channels, config.data.image_size, config.data.image_size, config.data.image_size)
        ).to(config.device)
    

        x0 = torch.zeros(sampling_shape, device=config.device)
        x0[0] = noise[0]
        x0[-1] = noise[1]
        for i in range(1, batch_size - 1):
            x[i] = slerp(x[0], x[-1], i / float(batch_size - 1))

        samples, n = sampling_fn(score_model, x0=x0)
        samples = samples.cpu().numpy()
        np.save(save_file_path, samples)


def cond_gen(
        config,
        save_fname='0',
    ):
    """
        Conditional Generation with partially completed dmtet from a 2.5D view (converted into a cubic grid)
    """
    with torch.no_grad():
        eval_dir, ckpt_path = config.eval.eval_dir, config.eval.ckpt_path
        # Create directory to eval_folder
        os.makedirs(eval_dir, exist_ok=True)

        scaler, inverse_scaler = lambda x: x, lambda x: x

        # Initialize model
        score_model = mutils.create_model(config)
        optimizer = losses.get_optimizer(config, score_model.parameters())
        ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

        # Setup SDEs
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)

        resolution = config.data.image_size
        grid_mask = torch.load(f'./data/grid_mask_{resolution}.pt').view(1, 1, resolution, resolution, resolution).to("cuda")

        sampling_eps = 1e-3
        sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        resolution, resolution, resolution)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, grid_mask=grid_mask)

        assert os.path.exists(ckpt_path)
        print('ckpt path:', ckpt_path)
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            raise
        ema.copy_to(score_model.parameters())

        print(f"loaded model is trained till iter {state['step'] // config.training.iter_size}")

        
        save_file_path = os.path.join(eval_dir, f"{save_fname}.npy")

        ### Conditional but free gradients; start from small t

        partial_dict = torch.load(config.eval.partial_dmtet_path)
        partial_sdf = partial_dict['sdf']
        partial_mask = partial_dict['vis']


        ### compute the mapping from tet indices to 3D cubic grid vertex indices
        tet_path = config.eval.tet_path
        tet = np.load(tet_path)
        vertices = torch.tensor(tet['vertices'])
        vertices_unique = vertices[:].unique()
        dx = vertices_unique[1] - vertices_unique[0]

        ind_to_coord = (torch.round(
            (vertices - vertices.min()) / dx)
        ).long()

        
        partial_sdf_grid = torch.zeros((1, 1, resolution, resolution, resolution))
        partial_sdf_grid[0, 0, ind_to_coord[:, 0], ind_to_coord[:, 1], ind_to_coord[:, 2]] = partial_sdf
        partial_mask_grid = torch.zeros((1, 1, resolution, resolution, resolution))
        partial_mask_grid[0, 0, ind_to_coord[:, 0], ind_to_coord[:, 1], ind_to_coord[:, 2]] = partial_mask.float()

        samples, n = sampling_fn(
            score_model, 
            partial=partial_sdf_grid.cuda(), 
            partial_mask=partial_mask_grid.cuda(), 
            freeze_iters=config.eval.freeze_iters
        )

        samples = samples.cpu().numpy()
        np.save(save_file_path, samples)

