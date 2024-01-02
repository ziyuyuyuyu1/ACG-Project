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
"""Common layers for defining score networks.
"""
import math
import string
from functools import partial
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from .normalization import ConditionalInstanceNorm3dPlus
from einops import rearrange, repeat
from inspect import isfunction

def get_act(config):
  """Get activation functions from the config file."""

  if config.model.nonlinearity.lower() == 'elu':
    return nn.ELU()
  elif config.model.nonlinearity.lower() == 'relu':
    return nn.ReLU()
  elif config.model.nonlinearity.lower() == 'lrelu':
    return nn.LeakyReLU(negative_slope=0.2)
  elif config.model.nonlinearity.lower() == 'swish':
    return nn.SiLU()
  else:
    raise NotImplementedError('activation function does not exist!')


def ncsn_conv1x1(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=0):
  """1x1 convolution. Same as NCSNv1/v2."""
  conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, dilation=dilation,
                   padding=padding)
  init_scale = 1e-10 if init_scale == 0 else init_scale
  conv.weight.data *= init_scale
  conv.bias.data *= init_scale
  return conv


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
  """Ported from JAX. """

  def _compute_fans(shape, in_axis=1, out_axis=0):
    receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
    fan_in = shape[in_axis] * receptive_field_size
    fan_out = shape[out_axis] * receptive_field_size
    return fan_in, fan_out

  def init(shape, dtype=dtype, device=device):
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
    if mode == "fan_in":
      denominator = fan_in
    elif mode == "fan_out":
      denominator = fan_out
    elif mode == "fan_avg":
      denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        "invalid mode for variance scaling initializer: {}".format(mode))
    variance = scale / denominator
    if distribution == "normal":
      return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
    elif distribution == "uniform":
      return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
    else:
      raise ValueError("invalid distribution for variance scaling initializer")

  return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')


class Dense(nn.Module):
  """Linear layer with `default_init`."""
  def __init__(self):
    super().__init__()


def ddpm_conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv


def ncsn_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with PyTorch initialization. Same as NCSNv1/NCSNv2."""
  init_scale = 1e-10 if init_scale == 0 else init_scale
  conv = nn.Conv3d(in_planes, out_planes, stride=stride, bias=bias,
                   dilation=dilation, padding=padding, kernel_size=3)
  conv.weight.data *= init_scale
  conv.bias.data *= init_scale
  return conv


def ddpm_conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv

def ddpm_conv5x5(in_planes, out_planes, stride=2, bias=True, dilation=1, init_scale=1., padding=2):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv3d(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv


def ddpm_conv5x5_transposed(in_planes, out_planes, stride=2, bias=True, dilation=1, init_scale=1., padding=2):
  """3x3 convolution with DDPM initialization."""
  conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding,
                   dilation=dilation, bias=bias, output_padding=(0, 1))
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv


def ddpm_conv6x6_transposed(in_planes, out_planes, stride=2, bias=True, dilation=1, init_scale=1., padding=2):
  """3x3 convolution with DDPM initialization."""
  conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=6, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  nn.init.zeros_(conv.bias)
  return conv


  ###########################################################################
  # Functions below are ported over from the NCSNv1/NCSNv2 codebase:
  # https://github.com/ermongroup/ncsn
  # https://github.com/ermongroup/ncsnv2
  ###########################################################################


class CRPBlock(nn.Module):
  def __init__(self, features, n_stages, act=nn.ReLU(), maxpool=True):
    super().__init__()
    self.convs = nn.ModuleList()
    for i in range(n_stages):
      self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))
    self.n_stages = n_stages
    if maxpool:
      self.pool = nn.MaxPool3d(kernel_size=5, stride=1, padding=2)
    else:
      self.pool = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)

    self.act = act

  def forward(self, x):
    x = self.act(x)
    path = x
    for i in range(self.n_stages):
      path = self.pool(path)
      path = self.convs[i](path)
      x = path + x
    return x


class CondCRPBlock(nn.Module):
  def __init__(self, features, n_stages, num_classes, normalizer, act=nn.ReLU()):
    super().__init__()
    self.convs = nn.ModuleList()
    self.norms = nn.ModuleList()
    self.normalizer = normalizer
    for i in range(n_stages):
      self.norms.append(normalizer(features, num_classes, bias=True))
      self.convs.append(ncsn_conv3x3(features, features, stride=1, bias=False))

    self.n_stages = n_stages
    self.pool = nn.AvgPool3d(kernel_size=5, stride=1, padding=2)
    self.act = act

  def forward(self, x, y):
    x = self.act(x)
    path = x
    for i in range(self.n_stages):
      path = self.norms[i](path, y)
      path = self.pool(path)
      path = self.convs[i](path)

      x = path + x
    return x


class RCUBlock(nn.Module):
  def __init__(self, features, n_blocks, n_stages, act=nn.ReLU()):
    super().__init__()

    for i in range(n_blocks):
      for j in range(n_stages):
        setattr(self, '{}_{}_conv'.format(i + 1, j + 1), ncsn_conv3x3(features, features, stride=1, bias=False))

    self.stride = 1
    self.n_blocks = n_blocks
    self.n_stages = n_stages
    self.act = act

  def forward(self, x):
    for i in range(self.n_blocks):
      residual = x
      for j in range(self.n_stages):
        x = self.act(x)
        x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

      x += residual
    return x


class CondRCUBlock(nn.Module):
  def __init__(self, features, n_blocks, n_stages, num_classes, normalizer, act=nn.ReLU()):
    super().__init__()

    for i in range(n_blocks):
      for j in range(n_stages):
        setattr(self, '{}_{}_norm'.format(i + 1, j + 1), normalizer(features, num_classes, bias=True))
        setattr(self, '{}_{}_conv'.format(i + 1, j + 1), ncsn_conv3x3(features, features, stride=1, bias=False))

    self.stride = 1
    self.n_blocks = n_blocks
    self.n_stages = n_stages
    self.act = act
    self.normalizer = normalizer

  def forward(self, x, y):
    for i in range(self.n_blocks):
      residual = x
      for j in range(self.n_stages):
        x = getattr(self, '{}_{}_norm'.format(i + 1, j + 1))(x, y)
        x = self.act(x)
        x = getattr(self, '{}_{}_conv'.format(i + 1, j + 1))(x)

      x += residual
    return x


class MSFBlock(nn.Module):
  def __init__(self, in_planes, features):
    super().__init__()
    assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
    self.convs = nn.ModuleList()
    self.features = features

    for i in range(len(in_planes)):
      self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))

  def forward(self, xs, shape):
    sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
    for i in range(len(self.convs)):
      h = self.convs[i](xs[i])
      h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
      sums += h
    return sums


class CondMSFBlock(nn.Module):
  def __init__(self, in_planes, features, num_classes, normalizer):
    super().__init__()
    assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

    self.convs = nn.ModuleList()
    self.norms = nn.ModuleList()
    self.features = features
    self.normalizer = normalizer

    for i in range(len(in_planes)):
      self.convs.append(ncsn_conv3x3(in_planes[i], features, stride=1, bias=True))
      self.norms.append(normalizer(in_planes[i], num_classes, bias=True))

  def forward(self, xs, y, shape):
    sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
    for i in range(len(self.convs)):
      h = self.norms[i](xs[i], y)
      h = self.convs[i](h)
      h = F.interpolate(h, size=shape, mode='bilinear', align_corners=True)
      sums += h
    return sums


class RefineBlock(nn.Module):
  def __init__(self, in_planes, features, act=nn.ReLU(), start=False, end=False, maxpool=True):
    super().__init__()

    assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
    self.n_blocks = n_blocks = len(in_planes)

    self.adapt_convs = nn.ModuleList()
    for i in range(n_blocks):
      self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))

    self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)

    if not start:
      self.msf = MSFBlock(in_planes, features)

    self.crp = CRPBlock(features, 2, act, maxpool=maxpool)

  def forward(self, xs, output_shape):
    assert isinstance(xs, tuple) or isinstance(xs, list)
    hs = []
    for i in range(len(xs)):
      h = self.adapt_convs[i](xs[i])
      hs.append(h)

    if self.n_blocks > 1:
      h = self.msf(hs, output_shape)
    else:
      h = hs[0]

    h = self.crp(h)
    h = self.output_convs(h)

    return h


class CondRefineBlock(nn.Module):
  def __init__(self, in_planes, features, num_classes, normalizer, act=nn.ReLU(), start=False, end=False):
    super().__init__()

    assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
    self.n_blocks = n_blocks = len(in_planes)

    self.adapt_convs = nn.ModuleList()
    for i in range(n_blocks):
      self.adapt_convs.append(
        CondRCUBlock(in_planes[i], 2, 2, num_classes, normalizer, act)
      )

    self.output_convs = CondRCUBlock(features, 3 if end else 1, 2, num_classes, normalizer, act)

    if not start:
      self.msf = CondMSFBlock(in_planes, features, num_classes, normalizer)

    self.crp = CondCRPBlock(features, 2, num_classes, normalizer, act)

  def forward(self, xs, y, output_shape):
    assert isinstance(xs, tuple) or isinstance(xs, list)
    hs = []
    for i in range(len(xs)):
      h = self.adapt_convs[i](xs[i], y)
      hs.append(h)

    if self.n_blocks > 1:
      h = self.msf(hs, y, output_shape)
    else:
      h = hs[0]

    h = self.crp(h, y)
    h = self.output_convs(h, y)

    return h


class ConvMeanPool(nn.Module):
  def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
    super().__init__()
    if not adjust_padding:
      conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
      self.conv = conv
    else:
      conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

      self.conv = nn.Sequential(
        nn.ZeroPad3d((1, 0, 1, 0)),
        conv
      )

  def forward(self, inputs):
    output = self.conv(inputs)
    output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                  output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


class MeanPoolConv(nn.Module):
  def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
    super().__init__()
    self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

  def forward(self, inputs):
    output = inputs
    output = sum([output[:, :, ::2, ::2], output[:, :, 1::2, ::2],
                  output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return self.conv(output)


class UpsampleConv(nn.Module):
  def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
    super().__init__()
    self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
    self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

  def forward(self, inputs):
    output = inputs
    output = torch.cat([output, output, output, output], dim=1)
    output = self.pixelshuffle(output)
    return self.conv(output)


class ConditionalResidualBlock(nn.Module):
  def __init__(self, input_dim, output_dim, num_classes, resample=1, act=nn.ELU(),
               normalization=ConditionalInstanceNorm3dPlus, adjust_padding=False, dilation=None):
    super().__init__()
    self.non_linearity = act
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.resample = resample
    self.normalization = normalization
    if resample == 'down':
      if dilation > 1:
        self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
        self.normalize2 = normalization(input_dim, num_classes)
        self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
        conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
      else:
        self.conv1 = ncsn_conv3x3(input_dim, input_dim)
        self.normalize2 = normalization(input_dim, num_classes)
        self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
        conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

    elif resample is None:
      if dilation > 1:
        conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
        self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
        self.normalize2 = normalization(output_dim, num_classes)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
      else:
        conv_shortcut = nn.Conv3d
        self.conv1 = ncsn_conv3x3(input_dim, output_dim)
        self.normalize2 = normalization(output_dim, num_classes)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim)
    else:
      raise Exception('invalid resample value')

    if output_dim != input_dim or resample is not None:
      self.shortcut = conv_shortcut(input_dim, output_dim)

    self.normalize1 = normalization(input_dim, num_classes)

  def forward(self, x, y):
    output = self.normalize1(x, y)
    output = self.non_linearity(output)
    output = self.conv1(output)
    output = self.normalize2(output, y)
    output = self.non_linearity(output)
    output = self.conv2(output)

    if self.output_dim == self.input_dim and self.resample is None:
      shortcut = x
    else:
      shortcut = self.shortcut(x)

    return shortcut + output


class ResidualBlock(nn.Module):
  def __init__(self, input_dim, output_dim, resample=None, act=nn.ELU(),
               normalization=nn.InstanceNorm3d, adjust_padding=False, dilation=1):
    super().__init__()
    self.non_linearity = act
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.resample = resample
    self.normalization = normalization
    if resample == 'down':
      if dilation > 1:
        self.conv1 = ncsn_conv3x3(input_dim, input_dim, dilation=dilation)
        self.normalize2 = normalization(input_dim)
        self.conv2 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
        conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
      else:
        self.conv1 = ncsn_conv3x3(input_dim, input_dim)
        self.normalize2 = normalization(input_dim)
        self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
        conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

    elif resample is None:
      if dilation > 1:
        conv_shortcut = partial(ncsn_conv3x3, dilation=dilation)
        self.conv1 = ncsn_conv3x3(input_dim, output_dim, dilation=dilation)
        self.normalize2 = normalization(output_dim)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim, dilation=dilation)
      else:
        # conv_shortcut = nn.Conv3d ### Something wierd here.
        conv_shortcut = partial(ncsn_conv1x1)
        self.conv1 = ncsn_conv3x3(input_dim, output_dim)
        self.normalize2 = normalization(output_dim)
        self.conv2 = ncsn_conv3x3(output_dim, output_dim)
    else:
      raise Exception('invalid resample value')

    if output_dim != input_dim or resample is not None:
      self.shortcut = conv_shortcut(input_dim, output_dim)

    self.normalize1 = normalization(input_dim)

  def forward(self, x):
    output = self.normalize1(x)
    output = self.non_linearity(output)
    output = self.conv1(output)
    output = self.normalize2(output)
    output = self.non_linearity(output)
    output = self.conv2(output)

    if self.output_dim == self.input_dim and self.resample is None:
      shortcut = x
    else:
      shortcut = self.shortcut(x)

    return shortcut + output


###########################################################################
# Functions below are ported over from the DDPM codebase:
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
###########################################################################

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
  half_dim = embedding_dim // 2
  # magic number 10000 is from transformers
  emb = math.log(max_positions) / (half_dim - 1)
  # emb = math.log(2.) / (half_dim - 1)
  emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
  # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
  # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
  emb = timesteps.float()[:, None] * emb[None, :]
  emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
  if embedding_dim % 2 == 1:  # zero pad
    emb = F.pad(emb, (0, 1), mode='constant')
  assert emb.shape == (timesteps.shape[0], embedding_dim)
  return emb


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 4, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 4, 1, 2, 3)


class AttnBlock(nn.Module):
  """Channel-wise self-attention block."""
  def __init__(self, channels):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=0.)

  def forward(self, x):
    B, C, D, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bcdhw,bckij->bdhwkij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, D, H, W, D * H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, D, H, W, D, H, W))
    h = torch.einsum('bdhwkij,bckij->bcdhw', w, v)
    h = self.NIN_3(h)
    return x + h

def zero_module(module):
  """
  Zero out the parameters of a module and return it.
  """
  for p in module.parameters():
    p.detach().zero_()
  return module

def exists(val):
  return val is not None

def uniq(arr):
  return{el: True for el in arr}.keys()

def default(val, d):
  if exists(val):
    return val
  return d() if isfunction(d) else d

# feedforward
class GEGLU(nn.Module):
  def __init__(self, dim_in, dim_out):
    super().__init__()
    self.proj = nn.Linear(dim_in, dim_out * 2)

  def forward(self, x):
    x, gate = self.proj(x).chunk(2, dim=-1)
    return x * F.gelu(gate)

class FeedForward(nn.Module):
  def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
    super().__init__()
    inner_dim = int(dim * mult)
    dim_out = default(dim_out, dim)
    project_in = nn.Sequential(
      nn.Linear(dim, inner_dim),
      nn.GELU()
    ) if not glu else GEGLU(dim, inner_dim)

    self.net = nn.Sequential(
      project_in,
      nn.Dropout(dropout),
      nn.Linear(inner_dim, dim_out)
    )

  def forward(self, x):
    return self.net(x)

class ClassCondition(nn.Module):
  """Condition."""
  def __init__(self, num_classes, class_emb_dim, d, h, w, class_ch=1):
    super().__init__()
    self.class_ch = class_ch
    self.d = d
    self.h = h
    self.w = w
    self.class_embed = nn.Embedding(num_classes, class_emb_dim)
    self.class_condition = nn.Sequential(
            nn.Linear(class_emb_dim, class_ch * d * h * w),
            nn.SiLU(),
      )

  def forward(self, label):
    label = label.to(torch.int64)
    x = self.class_embed(label)
    x = self.class_condition(x)
    # print(f"label {label.shape} x {x.shape}")
    x = torch.reshape(x, (x.shape[0], self.class_ch, self.d, self.h, self.w))
    return x
  

class ClassConditionAttn(nn.Module):
  """Condition for attention."""
  def __init__(self, num_classes, class_emb_dim, d, s=1):
    super().__init__()
    self.d = d
    self.s = s
    self.class_embed = nn.Embedding(num_classes, class_emb_dim)
    self.class_condition = nn.Sequential(
            nn.Linear(class_emb_dim, d * s),
            nn.SiLU(),
      )

  def forward(self, label):
    label = label.to(torch.int64)
    x = self.class_embed(label)
    x = self.class_condition(x)
    x = torch.reshape(x, (x.shape[0], self.s, self.d))
    return x 

class CrossAttention(nn.Module):
  def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
    super().__init__()
    inner_dim = dim_head * heads
    context_dim = default(context_dim, query_dim)

    self.scale = dim_head ** -0.5
    self.heads = heads

    self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
    self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
    self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

    self.to_out = nn.Sequential(
      nn.Linear(inner_dim, query_dim),
      nn.Dropout(dropout)
    )

  def forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
      mask = rearrange(mask, 'b ... -> b (...)')
      max_neg_value = -torch.finfo(sim.dtype).max
      mask = repeat(mask, 'b j -> (b h) () j', h=h)
      sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    out = torch.einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)

class BasicTransformerBlock(nn.Module):
  def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, num_classes=None, class_embed_dim=None, sequence_length=None, use_text_context=False):
    super().__init__()
    self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
    self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
    self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)
    self.norm3 = nn.LayerNorm(dim)
    
    self.use_text_context = use_text_context
    
    if self.use_text_context:
      self.textprocess = nn.Linear(768, context_dim)
    else:
      self.condprocess = ClassConditionAttn(num_classes, class_embed_dim, context_dim, sequence_length)
    # self.checkpoint = checkpoint

  # def forward(self, x, context=None):
  #   return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

  def forward(self, x, context=None):
    if self.use_text_context:
      context = self.textprocess(context)
    else:
      context = self.condprocess(context)
    x = self.attn1(self.norm1(x)) + x
    x = self.attn2(self.norm2(x), context=context) + x
    x = self.ff(self.norm3(x)) + x
    return x

class SpatialTransformer(nn.Module):
  """
  Transformer block for image-like data.
  First, project the input (aka embedding)
  and reshape to b, t, d.
  Then apply standard transformer action.
  Finally, reshape to image
  """
  def __init__(self, channels, n_heads, d_head,
                depth=1, dropout=0., context_dim=None, num_classes=None, class_embed_dim=None, sequence_length=None, use_text_context=False):
    super().__init__()
    self.in_channels = channels
    inner_dim = n_heads * d_head
    self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)

    self.proj_in = NIN(channels, inner_dim)

    self.transformer_blocks = nn.ModuleList(
      [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                             num_classes=num_classes, class_embed_dim=class_embed_dim, sequence_length=sequence_length, use_text_context=use_text_context)
        for d in range(depth)]
    )

    self.proj_out = zero_module(NIN(inner_dim, channels))

  def forward(self, x, context=None):
    # note: if no context is given, cross-attention defaults to self-attention
    b, c, d, h, w = x.shape
    x_in = x
    x = self.norm(x)
    x = self.proj_in(x)
    x = rearrange(x, 'b c d h w -> b (d h w) c')
    for block in self.transformer_blocks:
      x = block(x, context=context)
    x = rearrange(x, 'b (d h w) c -> b c d h w', d=d, h=h, w=w)
    x = self.proj_out(x)
    return x + x_in

class Upsample(nn.Module):
  def __init__(self, channels, with_conv=False):
    super().__init__()
    if with_conv:
      self.Conv_0 = ddpm_conv3x3(channels, channels)
    self.with_conv = with_conv

  def forward(self, x):
    B, C, D, H, W = x.shape
    h = F.interpolate(x, (D * 2, H * 2, W * 2), mode='nearest')
    if self.with_conv:
      h = self.Conv_0(h)
    return h


class Downsample(nn.Module):
  def __init__(self, channels, with_conv=False):
    super().__init__()
    if with_conv:
      self.Conv_0 = ddpm_conv3x3(channels, channels, stride=2, padding=0)
    self.with_conv = with_conv

  def forward(self, x):
    B, C, D, H, W = x.shape
    # Emulate 'SAME' padding
    if self.with_conv:
      x = F.pad(x, (0, 1, 0, 1, 0, 1))
      x = self.Conv_0(x)
    else:
      x = F.avg_pool3d(x, kernel_size=2, stride=2, padding=0)

    assert x.shape == (B, C, D // 2, H // 2, W // 2)
    return x


class ResnetBlockDDPM(nn.Module):
  """The ResNet Blocks used in DDPM."""
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1, additional_dim=0):
    super().__init__()
    if out_ch is None:
      out_ch = in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
    self.additional_dim = additional_dim
    self.act = act
    self.Conv_0 = ddpm_conv3x3(in_ch+additional_dim, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.)
    if in_ch + additional_dim != out_ch:
      if conv_shortcut:
        self.Conv_2 = ddpm_conv3x3(in_ch + additional_dim, out_ch)
      else:
        self.NIN_0 = NIN(in_ch + additional_dim, out_ch)
    self.out_ch = out_ch
    self.in_ch = in_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    B, C, D, H, W = x.shape
    assert C == self.in_ch + self.additional_dim
    out_ch = self.out_ch if self.out_ch else self.in_ch
    y = self.GroupNorm_0(x[:,0:self.in_ch])
    y = torch.cat([y, x[:,self.in_ch:]], dim=1)
    h = self.act(y)
    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if C != out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    return x + h

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

class ResnetBlockDDPMPosEncoding(nn.Module):
  """The ResNet Blocks used in DDPM."""
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1, img_size=64):
    super().__init__()
    ##### Pos Encoding
    coord_x, coord_y, coord_z = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), torch.arange(img_size))
    coords = torch.stack([coord_x, coord_y, coord_z])
    self.num_freq = int(np.log2(img_size))
    pos_encoding = torch.zeros(1, 2 * self.num_freq, 3, img_size, img_size, img_size)
    with torch.no_grad():
      for i in range(self.num_freq):
        pos_encoding[0, 2*i,     :, :, :, :] = torch.cos((i+1) * np.pi * coords)
        pos_encoding[0, 2*i + 1, :, :, :, :] = torch.sin((i+1) * np.pi * coords)
    self.pos_encoding = nn.Parameter(
      pos_encoding.view(1, 2 * self.num_freq * 3, img_size, img_size, img_size) / img_size,
      requires_grad=False
    )
    ####

    if out_ch is None:
      out_ch = in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6)
    self.act = act
    self.Conv_0 = ddpm_conv3x3(in_ch, out_ch)
    self.Conv_0_pos = ddpm_conv3x3(2 * self.num_freq * 3, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = ddpm_conv3x3(out_ch, out_ch, init_scale=0.)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = ddpm_conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)
    self.out_ch = out_ch
    self.in_ch = in_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    B, C, D, H, W = x.shape
    assert C == self.in_ch
    out_ch = self.out_ch if self.out_ch else self.in_ch
    h = self.act(self.GroupNorm_0(x))
    h = self.Conv_0(h) + self.Conv_0_pos(self.pos_encoding).expand(h.size(0), -1, -1, -1, -1)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if C != out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    return x + h

