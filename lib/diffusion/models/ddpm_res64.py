# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import torch
import torch.nn as nn
import functools
import numpy as np

from . import utils, layers, normalization

# RefineBlock = layers.RefineBlock
# ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
  
  

@utils.register_model(name='ddpm_res64')
class DDPMRes64(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]
    
    self.use_spatial = use_spatial = config.model.use_spatial
    if use_spatial:
      num_heads = config.model.num_heads
      dim_head = config.model.dim_head
      transformer_depth = config.model.transformer_depth
      context_dim = config.model.context_dim

    self.context_concat = context_concat = config.model.context_concat
    if context_concat:
      context_concat_dim = config.model.context_concat_dim
      num_classes = config.model.num_classes # 8
      class_emb_dim = config.model.class_emb_dim

    self.use_text_context = use_text_context = config.model.use_text_context
    
    if context_concat:
      ClassCondition = functools.partial(layers.ClassCondition, num_classes, class_emb_dim, class_ch=context_concat_dim)

    AttnBlock = functools.partial(layers.AttnBlock) if not use_spatial else functools.partial(layers.SpatialTransformer, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, context_dim=context_dim, num_classes=num_classes, class_embed_dim=class_emb_dim, sequence_length=context_concat_dim, use_text_context=use_text_context)
    self.conditional = conditional = config.model.conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = config.data.centered
    channels = config.data.num_channels


    ##### Pos Encoding
    self.img_size = img_size = config.data.image_size
    self.num_freq = int(np.log2(img_size))
    coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), torch.arange(img_size))
    self.coords = torch.nn.Parameter(
      torch.stack([coord_x, coord_y, coord_z]).view(1, 3, img_size, img_size, img_size) * 0.0,
      requires_grad=False
    )
    ####

    class_size = img_size

    ### Mask
    self.mask = torch.nn.Parameter(torch.zeros(1, 1, img_size, img_size, img_size), requires_grad=False)

    # Downsampling block
    self.pos_layer = conv3x3(3, nf)
    self.mask_layer = conv3x3(1, nf)
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        if not context_concat:
          modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        else:
          modules.append(ClassCondition(d=class_size, h=class_size, w=class_size))
          modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch, additional_dim=context_concat_dim))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)
        class_size = class_size // 2

    in_ch = hs_c[-1]
    # modules.append(ResnetBlock(in_ch=in_ch))
    if not context_concat:
      modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
    else:
      modules.append(ClassCondition(d=class_size, h=class_size, w=class_size))
      modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch, additional_dim=context_concat_dim))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        # modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        if not context_concat:
          modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
        else:
          modules.append(ClassCondition(d=class_size, h=class_size, w=class_size))
          modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch, additional_dim=context_concat_dim))
        in_ch = out_ch
      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))
      if i_level != 0:
        modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))
        class_size = class_size * 2

    assert not hs_c
    modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
    modules.append(conv3x3(in_ch, channels, init_scale=0.))
    self.all_modules = nn.ModuleList(modules)

    self.scale_by_sigma = config.model.scale_by_sigma

  def forward(self, x, labels, context=None, use_text_context=False):
    # print(context)
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block    
    hs = [modules[m_idx](h) + self.pos_layer(self.coords) + self.mask_layer(self.mask)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = hs[-1]
        if self.context_concat:
          cond = modules[m_idx](context)
          h = torch.concat([h, cond], dim=1)
          m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          if not self.use_spatial:
            h = modules[m_idx](h)
          else:
            h = modules[m_idx](h, context)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    if self.context_concat:
      cond = modules[m_idx](context)
      h = torch.concat([h, cond], dim=1)
      m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1
    if not self.use_spatial:
      h = modules[m_idx](h)
    else:
      h = modules[m_idx](h, context)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        hspop = hs.pop()
        h = torch.cat([h, hspop], dim=1)
        if self.context_concat:
          cond = modules[m_idx](context)
          h = torch.concat([h, cond], dim=1)
          m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1
      if h.shape[-1] in self.attn_resolutions:
        if not self.use_spatial:
          h = modules[m_idx](h)
        else:
          h = modules[m_idx](h, context)
        m_idx += 1
      if i_level != 0:
        h = modules[m_idx](h)
        m_idx += 1

    assert not hs
    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    assert m_idx == len(modules)

    if self.scale_by_sigma:
      # Divide the output by sigmas. Useful for training with the NCSN loss.
      # The DDPM loss scales the network output by sigma in the loss function,
      # so no need of doing it here.
      used_sigmas = self.sigmas[labels, None, None, None]
      h = h / used_sigmas

    return h


@utils.register_model(name='ddpm_res64_encode')
class DDPMRes64Encode(nn.Module):
  def __init__(self, nf, ch_mult, num_res_blocks, attn_resolutions, resamp_with_conv, conditional, dropout, image_size, data, **kwargs):
    super().__init__()
    self.act = act = nn.SiLU()

    self.nf = nf
    self.num_res_blocks = num_res_blocks
    self.attn_resolutions = attn_resolutions
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]
    

    AttnBlock = functools.partial(layers.AttnBlock)
    self.conditional = conditional
    ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout)
    if conditional:
      # Condition on noise levels.
      modules = [nn.Linear(nf, nf * 4)]
      modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
      nn.init.zeros_(modules[0].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
      nn.init.zeros_(modules[1].bias)

    self.centered = data['centered']
    channels = data['num_channels']


    ##### Pos Encoding
    self.img_size = img_size = image_size
    self.num_freq = int(np.log2(img_size))
    coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), torch.arange(img_size))
    self.coords = torch.nn.Parameter(
      torch.stack([coord_x, coord_y, coord_z]).view(1, 3, img_size, img_size, img_size) * 0.0,
      requires_grad=False
    )
    ####

    class_size = img_size

    ### Mask
    self.mask = torch.nn.Parameter(torch.zeros(1, 1, img_size, img_size, img_size), requires_grad=False)

    # Downsampling block
    self.pos_layer = conv3x3(3, nf)
    self.mask_layer = conv3x3(1, nf)
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]
    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch
        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)
      if i_level != num_resolutions - 1:
        modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
        hs_c.append(in_ch)
        class_size = class_size // 2

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))
    
    self.out = nn.Sequential(
                nn.GroupNorm(32, in_ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                zero_module(nn.Conv3d(in_ch, 5, 1)),
                nn.Flatten(),
            )
    
    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, labels, context=None, use_text_context=False):
    # print(context)
    modules = self.all_modules
    m_idx = 0
    if self.conditional:
      # timestep/scale embedding
      timesteps = labels
      temb = layers.get_timestep_embedding(timesteps, self.nf)
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.centered:
      # Input is in [-1, 1]
      h = x
    else:
      # Input is in [0, 1]
      h = 2 * x - 1.

    # Downsampling block    
    hs = [modules[m_idx](h) + self.pos_layer(self.coords) + self.mask_layer(self.mask)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1
        hs.append(h)
      if i_level != self.num_resolutions - 1:
        hs.append(modules[m_idx](hs[-1]))
        m_idx += 1

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1
    
    h = self.out(h)

    return h