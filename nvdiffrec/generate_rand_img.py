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

# # Import data readers / generators
# from lib.dataset.dataset_mesh import DatasetMesh
# from lib.dataset.dataset_shapenet import ShapeNetDataset

# # Import topology / geometry trainers
# from lib.geometry.dmtet import DMTetGeometry
# from lib.geometry.dmtet_fixedtopo import DMTetGeometryFixedTopo

from lib.render import util


RADIUS = 2.0

def gen(glctx, geometry, opt_material, lgt, FLAGS=argparse.ArgumentParser()):
    #-------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------    
    result_dict = {}
    proj_mtx = util.perspective(np.deg2rad(45), FLAGS.display_res[1] / FLAGS.display_res[0], FLAGS.cam_near_far[0], FLAGS.cam_near_far[1])
    ang  =  np.pi * 2 * np.random.rand()
    mv = util.translate(0, 0, - RADIUS) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
    mvp = proj_mtx @ mv
    campos = torch.linalg.inv(mv)[:3, 3]
    target = {
        'mv': mv[None, ...].cuda(),
        'envlight_transform': None,
        'mvp': mvp[None, ...].cuda(),
        'campos': campos[None, ...].cuda(),
        'resolution': FLAGS.display_res,
        'spp': FLAGS.spp,
        'background': torch.ones([4, 1000, 1000, 3], dtype=torch.float32, device='cuda')
    }
    # with torch.no_grad():
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
    
    ang = ang * 180 / np.pi - 180
    
    return ang, result_dict


