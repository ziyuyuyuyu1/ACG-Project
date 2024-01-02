import torch
import sys
sys.path.append("/home/jialuo/CG_project")
from MeshDiffusion.lib.diffusion.models.ddpm_res64 import DDPMRes64Encode


if __name__ == "__main__":
    device = torch.device("cuda")
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
    
    unet_model = DDPMRes64Encode(**model_config).to(device)
    
    count = 0
    for p in unet_model.parameters():
        count += p.numel()
    print("Param Count: ", count)

    x = torch.randn(1, 4, 64, 64, 64).to(device)
    t = torch.randint(0, 1000, (x.shape[0], ), device=device).long()
    out = unet_model(x, t)
    print(out.shape)