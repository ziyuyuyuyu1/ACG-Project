import os
import sys
import torch
from torch.utils.data import Dataset
import json
import numpy as np
import torch.nn as nn
sys.path.append('/home/jialuo/CG_project/MeshDiffusion')
from binvox_rw import binvox_rw
import argparse

class ShapeNetVoxelDataset(Dataset):
    def __init__(self, root, grid_mask, deform_scale=1.0, aug=False, filter_meta_path=None, normalize_sdf=True, extension='pt'):
        super().__init__()
        self.fpath_list = json.load(open(root, 'r'))
        self.aug = aug
        self.grid_mask = grid_mask.cpu()
        self.resolution = self.grid_mask.size(-1)
    
    def __len__(self):
        return len(self.fpath_list)

    def __getitem__(self, idx):
        with torch.no_grad():
            with open(self.fpath_list[idx], 'rb') as f:
                datum = torch.from_numpy(binvox_rw.read_as_3d_array(f).data.astype(np.float32)).unsqueeze(0)
                
            avg_pool = nn.AvgPool3d(2)
            datum = avg_pool(datum)
            # if self.aug:
            #     nonempty_mask = (datum[1:].abs().sum(dim=0, keepdim=True) != 0)
            #     datum[1:] = datum[1:] + (torch.rand(3)[:, None, None, None] - 0.5) * 0.01 * nonempty_mask / (datum.size(-1) / self.resolution)

            #     if datum.size(-1) < self.resolution:
            #         datum = datum * self.grid_mask[0, :, :datum.size(-1), :datum.size(-1), :datum.size(-1)]
            #     else:
            #         datum = datum * self.grid_mask[0]

        
        if datum.size(-1) < self.resolution: 
            diff = self.resolution - datum.size(-1)
            datum = torch.nn.functional.pad(datum, (0, diff, 0, diff, 0, diff, 0, 0))
            
        
        
        return datum
