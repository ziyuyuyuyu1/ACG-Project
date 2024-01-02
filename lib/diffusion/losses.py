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

"""All functions related to loss computation and optimization.
"""
import sys
sys.path.append('/home/jialuo/CG_project/MeshDiffusion')
import torch
import torch.optim as optim
import numpy as np
from .models import utils as mutils
from .sde_lib import VPSDE
# from nvdiffrec import test
import torch.nn.functional as F

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn

def get_ddpm_loss_fn(vpsde, train, mask=None, loss_type='l2', classifier=False):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""

  def loss_fn(model, batch, twod_guide=False, use_text_context=False, sd_model=None, class_conditioned=True):
    model_fn = mutils.get_model_fn(model, train=train)
    if twod_guide:
      text = batch
    elif use_text_context:
      context = batch['text_feat']
      batch = batch['datum']
    elif classifier:
      batch, context = batch.chunk(2, dim=-1)
    elif class_conditioned:
      batch, context = batch.chunk(2, dim=-1)
    else:
      context = None
      mask = None
      batch = batch
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    if not twod_guide:
      perturbed_data = sqrt_alphas_cumprod[labels, None, None, None, None] * batch + \
                      sqrt_1m_alphas_cumprod[labels, None, None, None, None] * noise
    else:
      # pertubed_data is pure noise
      ones = torch.ones_like(sqrt_1m_alphas_cumprod)
      perturbed_data = ones[labels, None, None, None, None] * noise
    if mask is not None:
      perturbed_data = perturbed_data * mask
    if use_text_context:
      score = model_fn(perturbed_data, labels, context, use_text_context=True)
    elif classifier:
      logits = model_fn(perturbed_data, labels)
      classes = (context[:,0,0,0,0] - 1.).long()
      losses = F.cross_entropy(logits, classes, reduction="none")
      loss = torch.mean(losses)
      return loss
    elif context is not None:
      score = model_fn(perturbed_data, labels, context[:,0,0,0,0].unsqueeze(-1))
    else:
      score = model_fn(perturbed_data, labels, context)
      
    if not twod_guide:
      if loss_type == 'l2':
        losses = torch.square(score - noise)
      elif loss_type == 'l1':
        losses = torch.abs(score - noise)
      else:
        raise NotImplementedError
    else:
      # x0_predict = (perturbed_data - score * sqrt_1m_alphas_cumprod[labels, None, None, None, None]) / sqrt_alphas_cumprod[labels, None, None, None, None]
      # # generate image
      # img_predict, angle = test.generate_img(x0_predict)
      # # get text embedding
      # text_embedding = sd_model.get_text_all_embeddings(text, '', angle)
      # # compute loss
      # losses = sd_model.train_step(text_embedding, x0_predict)
      raise NotImplementedError
      
      

    if mask is not None:
      losses = losses * mask
      losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
      loss = torch.mean(losses) / mask.sum() * np.prod(mask.size())
    else:
      losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
      loss = torch.mean(losses)

    return loss

  return loss_fn

def get_step_fn(sde, train, optimize_fn=None, mask=None, loss_type='l2'):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """

  loss_fn = get_ddpm_loss_fn(sde, train,  mask=mask, loss_type=loss_type)

  def step_fn(state, batch, clear_grad=True, update_param=True, twod_guide=False, use_text_context=False, sd_model=None, class_conditioned=True):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      if clear_grad:
        optimizer.zero_grad()
      loss = loss_fn(model, batch, twod_guide=twod_guide, use_text_context=use_text_context, sd_model=sd_model, class_conditioned=class_conditioned)
      loss.backward()
      if update_param:
        optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return {
      'loss': loss,
    }

  return step_fn

def classifier_get_step_fn(sde, train, optimize_fn=None, mask=None, loss_type='l2'):
  
  loss_fn = get_ddpm_loss_fn(sde, train,  mask=mask, loss_type=loss_type, classifier=True)
  
  def classifier_step_fn(state, batch, clear_grad=True, update_param=True, twod_guide=False, use_text_context=False, sd_model=None):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
      EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    
    
    
    model = state['model']
    if train:
      optimizer = state['optimizer']
      if clear_grad:
        optimizer.zero_grad()
      loss = loss_fn(model, batch, twod_guide=twod_guide, use_text_context=use_text_context, sd_model=sd_model)
      
      loss.backward()
      if update_param:
        optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      # with torch.no_grad():
      #   ema = state['ema']
      #   ema.store(model.parameters())
      #   ema.copy_to(model.parameters())
      #   loss = loss_fn(model, batch)
      #   ema.restore(model.parameters())
      raise NotImplementedError

    return {
      'loss': loss,
      }

  return classifier_step_fn