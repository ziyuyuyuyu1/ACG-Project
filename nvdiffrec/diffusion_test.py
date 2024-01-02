import sys
sys.path.append('/home/jialuo/CG_project/MeshDiffusion')
from nvdiffrec.sd_utils import StableDiffusion
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

sd = StableDiffusion(device='cuda')
neg = ''
text = 'a sphere'

test = sd.get_text_all_embeddings(text, neg, torch.zeros((1,1)))
print(test.shape)