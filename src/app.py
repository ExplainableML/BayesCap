import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode

from PIL import Image

from ds import *
from losses import *
from networks_SRGAN import *
from utils import *


NetG = Generator()
model_parameters = filter(lambda p: True, NetG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of Parameters:",params)
NetC = BayesCap(in_channels=3, out_channels=3)


NetG = Generator()
NetG.load_state_dict(torch.load('../ckpt/srgan-ImageNet-bc347d67.pth', map_location='cuda:0'))
NetG.to('cuda')
NetG.eval()

NetC = BayesCap(in_channels=3, out_channels=3)
NetC.load_state_dict(torch.load('../ckpt/BayesCap_SRGAN_best.pth', map_location='cuda:0'))
NetC.to('cuda')
NetC.eval()

def tensor01_to_pil(xt):
    r = transforms.ToPILImage(mode='RGB')(xt.squeeze())
    return r


def predict(img):
    """
    img: image
    """
    image_size = (256,256)
    upscale_factor = 4
    lr_transforms = transforms.Resize((image_size[0]//upscale_factor, image_size[1]//upscale_factor), interpolation=IMode.BICUBIC, antialias=True)
    # lr_transforms = transforms.Resize((128, 128), interpolation=IMode.BICUBIC, antialias=True)
    
    img = Image.fromarray(np.array(img))
    img = lr_transforms(img)
    lr_tensor = utils.image2tensor(img, range_norm=False, half=False)
    
    device = 'cuda'
    dtype = torch.cuda.FloatTensor
    xLR = lr_tensor.to(device).unsqueeze(0)
    xLR = xLR.type(dtype)
    # pass them through the network
    with torch.no_grad():
        xSR = NetG(xLR)
        xSRC_mu, xSRC_alpha, xSRC_beta = NetC(xSR)
        
    a_map = (1/(xSRC_alpha[0] + 1e-5)).to('cpu').data
    b_map = xSRC_beta[0].to('cpu').data
    u_map = (a_map**2)*(torch.exp(torch.lgamma(3/(b_map + 1e-2)))/torch.exp(torch.lgamma(1/(b_map + 1e-2)))) 
    
    
    x_LR = tensor01_to_pil(xLR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1))
    
    x_mean = tensor01_to_pil(xSR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1))
    
    #im = Image.fromarray(np.uint8(cm.gist_earth(myarray)*255))
    
    a_map = torch.clamp(a_map, min=0, max=0.1)
    a_map = (a_map - a_map.min())/(a_map.max() - a_map.min())
    x_alpha = Image.fromarray(np.uint8(cm.inferno(a_map.transpose(0,2).transpose(0,1).squeeze())*255))
    
    b_map = torch.clamp(b_map, min=0.45, max=0.75)
    b_map = (b_map - b_map.min())/(b_map.max() - b_map.min())
    x_beta = Image.fromarray(np.uint8(cm.cividis(b_map.transpose(0,2).transpose(0,1).squeeze())*255))
    
    u_map = torch.clamp(u_map, min=0, max=0.15)
    u_map = (u_map - u_map.min())/(u_map.max() - u_map.min())
    x_uncer = Image.fromarray(np.uint8(cm.hot(u_map.transpose(0,2).transpose(0,1).squeeze())*255))
    
    return x_LR, x_mean, x_alpha, x_beta, x_uncer

import gradio as gr

title = "BayesCap"
description = "BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks (ECCV 2022)"
article = "<p style='text-align: center'> BayesCap: Bayesian Identity Cap for Calibrated Uncertainty in Frozen Neural Networks| <a href='https://github.com/ExplainableML/BayesCap'>Github Repo</a></p>"


gr.Interface(
    fn=predict, 
    inputs=gr.inputs.Image(type='pil', label="Orignal"), 
    outputs=[
        gr.outputs.Image(type='pil', label="Low-res"), 
        gr.outputs.Image(type='pil', label="Super-res"), 
        gr.outputs.Image(type='pil', label="Alpha"), 
        gr.outputs.Image(type='pil', label="Beta"), 
        gr.outputs.Image(type='pil', label="Uncertainty")
     ],
    title=title,
    description=description,
    article=article,
     examples=[
        ["../demo_examples/baby.png"],
        ["../demo_examples/bird.png"]
    ]
).launch(share=True)