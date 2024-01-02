"""Config file for reproducing the results of DDPM on bedrooms."""

from configs.default_configs import get_default_configs
import torch

def get_config():
  config = get_default_configs()
  
  # config.device = torch.device('cuda:5')

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = False
  training.reduce_mean = True
  training.batch_size = 10
  training.lip_scale = None
  training.text_context = False

  training.snapshot_freq_for_preemption = 1000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'ancestral_sampling'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.dataset = 'ShapeNet'
  data.centered = True
  data.image_size = 64
  data.num_channels = 4
  data.meta_path = "PLACEHOLDER" ### metadata for all dataset files
  data.filter_meta_path = "PLACEHOLDER" ### metadata for the list of training samples
  data.num_workers = 4
  data.aug = True


  # model
  model = config.model
  model.name = 'ddpm_res64'
  model.scale_by_sigma = False
  model.num_scales = 1000
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 4, 4)
  model.num_res_blocks_first = 2
  model.num_res_blocks = 3
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.dropout = 0.1

  # spatial
  model.use_spatial = True
  model.num_heads = 8
  model.dim_head = 64
  model.transformer_depth = 1
  model.context_dim = 128

  # context_concat
  model.context_concat = False
  model.context_concat_dim = 1
  model.num_classes = 8
  model.class_emb_dim = 64

  # optim
  optim = config.optim
  optim.lr = 2e-5

  config.eval.batch_size = 5
  config.eval.eval_dir = "PLACEHOLDER"

  config.seed = 42

  return config
