import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 64
  training.n_iters = 90000
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.iter_size = 1
  training.loss_type = 'l2'
  training.train_dir = "/share1/jialuo/PLACEHOLDER111"
  
  training.twod_guide = False
  training.text_context = False
  
  training.classifier = False
  
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 50
  evaluate.end_ckpt = 96
  evaluate.batch_size = 512
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.ckpt_path = "PLACEHOLDER"
  evaluate.partial_dmtet_path = "PLACEHOLDER"
  evaluate.tet_path = "PLACEHOLDER"
  evaluate.freeze_iters = 950

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'LSUN'
  data.image_size = 256
  data.random_flip = True
  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 3
  data.num_workers = 4
  data.normalize_sdf = True
  data.meta_path = "PLACEHOLDER" ### metadata for all dataset files
  data.filter_meta_path = "PLACEHOLDER" ### metadata for the list of training samples
  data.extension = 'pt' ### either 'pt' or 'npy', depending how the data are stored

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 378
  model.sigma_min = 0.01
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.
  model.embedding_type = 'fourier'
  model.deform_scale = 1.0
  model.use_spatial = False
  model.use_text_context = False
  
  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  config.mask = True


  # rendering
  config.render = render = ml_collections.ConfigDict()
  config.eval.classifier_scale = 10.

  return config