from imageio import imread
import torch
img = imread('/home/jialuo/CG_project/MeshDiffusion/nvdiffrec/img_tinted5.png')
print(img.shape)
print(img.dtype)
t_img = torch.tensor(img)
x_img = t_img.float() / 255.0
print(x_img.dtype)