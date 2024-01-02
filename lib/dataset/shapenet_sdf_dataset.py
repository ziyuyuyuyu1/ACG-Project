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
import trimesh
import mesh2sdf

class ShapeNetSdfDataset(Dataset):
    def __init__(self, root, grid_mask, deform_scale=1.0, aug=False, filter_meta_path=None, normalize_sdf=True, extension='pt'):
        super().__init__()
        self.fpath_list = json.load(open(root, 'r'))
        self.aug = aug
        self.grid_mask = grid_mask.cpu()
        self.resolution = self.grid_mask.size(-1)
    
    def __len__(self):
        return len(self.fpath_list)

    def __getitem__(self, idx):
        mesh_scale = 0.8
        size = 64
        level = 2 / size
        with torch.no_grad():
            mesh = trimesh.load(self.fpath_list[idx], force='mesh')
            vertices = mesh.vertices
            bbmin = vertices.min(0)
            bbmax = vertices.max(0)
            center = (bbmin + bbmax) * 0.5
            scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
            vertices = (vertices - center) * scale
            datum = mesh2sdf.compute(
                    vertices, mesh.faces, size, fix=True, level=level, return_mesh=False)
            datum = torch.from_numpy(datum.astype(np.float32)).unsqueeze(0)
        
        if datum.size(-1) < self.resolution: 
            diff = self.resolution - datum.size(-1)
            datum = torch.nn.functional.pad(datum, (0, diff, 0, diff, 0, diff, 0, 0))
        
        return datum
