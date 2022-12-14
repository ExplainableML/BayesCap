import random
from typing import Any, Optional
import numpy as np
import os
import cv2
from glob import glob
from PIL import Image, ImageDraw
from tqdm import tqdm
import kornia
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as albu
import functools
import math

import torch
import torch.nn as nn
from torch import Tensor
import torchvision as tv
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import functional as F
from losses import TempCombLoss

######## for loading checkpoint from googledrive
google_drive_paths = {
    "BayesCap_SRGAN.pth": "https://drive.google.com/uc?id=1d_5j1f8-vN79htZTfRUqP1ddHZIYsNvL",
    "BayesCap_ckpt.pth": "https://drive.google.com/uc?id=1Vg1r6gKgQ1J3M51n6BeKXYS8auT9NhA9",
}

def ensure_checkpoint_exists(model_weights_filename):
    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename in google_drive_paths
    ):
        gdrive_url = google_drive_paths[model_weights_filename]
        try:
            from gdown import download as drive_download

            drive_download(gdrive_url, model_weights_filename, quiet=False)
        except ModuleNotFoundError:
            print(
                "gdown module not found.",
                "pip3 install gdown or, manually download the checkpoint file:",
                gdrive_url
            )

    if not os.path.isfile(model_weights_filename) and (
        model_weights_filename not in google_drive_paths
    ):
        print(
            model_weights_filename,
            " not found, you may need to manually download the model weights."
        )

########### DeblurGAN function
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process

def preprocess(x: np.ndarray, mask: Optional[np.ndarray]):
    x, _ = get_normalize()(x, x)
    if mask is None:
        mask = np.ones_like(x, dtype=np.float32)
    else:
        mask = np.round(mask.astype('float32') / 255)

    h, w, _ = x.shape
    block_size = 32
    min_height = (h // block_size + 1) * block_size
    min_width = (w // block_size + 1) * block_size

    pad_params = {'mode': 'constant',
                    'constant_values': 0,
                    'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                    }
    x = np.pad(x, **pad_params)
    mask = np.pad(mask, **pad_params)

    return map(_array_to_batch, (x, mask)), h, w

def postprocess(x: torch.Tensor) -> np.ndarray:
    x, = x
    x = x.detach().cpu().float().numpy()
    x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
    return x.astype('uint8')

def sorted_glob(pattern):
    return sorted(glob(pattern))
###########

def normalize(image: np.ndarray) -> np.ndarray:
	"""Normalize the ``OpenCV.imread`` or ``skimage.io.imread`` data.
	Args:
		image (np.ndarray): The image data read by ``OpenCV.imread`` or ``skimage.io.imread``.
	Returns:
		Normalized image data. Data range [0, 1].
	"""
	return image.astype(np.float64) / 255.0


def unnormalize(image: np.ndarray) -> np.ndarray:
	"""Un-normalize the ``OpenCV.imread`` or ``skimage.io.imread`` data.
	Args:
		image (np.ndarray): The image data read by ``OpenCV.imread`` or ``skimage.io.imread``.
	Returns:
		Denormalized image data. Data range [0, 255].
	"""
	return image.astype(np.float64) * 255.0


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
	"""Convert ``PIL.Image`` to Tensor.
	Args:
		image (np.ndarray): The image data read by ``PIL.Image``
		range_norm (bool): Scale [0, 1] data to between [-1, 1]
		half (bool): Whether to convert torch.float32 similarly to torch.half type.
	Returns:
		Normalized image data
	Examples:
		>>> image = Image.open("image.bmp")
		>>> tensor_image = image2tensor(image, range_norm=False, half=False)
	"""
	tensor = F.to_tensor(image)

	if range_norm:
		tensor = tensor.mul_(2.0).sub_(1.0)
	if half:
		tensor = tensor.half()

	return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
	"""Converts ``torch.Tensor`` to ``PIL.Image``.
	Args:
		tensor (torch.Tensor): The image that needs to be converted to ``PIL.Image``
		range_norm (bool): Scale [-1, 1] data to between [0, 1]
		half (bool): Whether to convert torch.float32 similarly to torch.half type.
	Returns:
		Convert image data to support PIL library
	Examples:
		>>> tensor = torch.randn([1, 3, 128, 128])
		>>> image = tensor2image(tensor, range_norm=False, half=False)
	"""
	if range_norm:
		tensor = tensor.add_(1.0).div_(2.0)
	if half:
		tensor = tensor.half()

	image = tensor.squeeze_(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).cpu().numpy().astype("uint8")

	return image


def convert_rgb_to_y(image: Any) -> Any:
	"""Convert RGB image or tensor image data to YCbCr(Y) format.
	Args:
		image: RGB image data read by ``PIL.Image''.
	Returns:
		Y image array data.
	"""
	if type(image) == np.ndarray:
		return 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
	elif type(image) == torch.Tensor:
		if len(image.shape) == 4:
			image = image.squeeze_(0)
		return 16. + (64.738 * image[0, :, :] + 129.057 * image[1, :, :] + 25.064 * image[2, :, :]) / 256.
	else:
		raise Exception("Unknown Type", type(image))


def convert_rgb_to_ycbcr(image: Any) -> Any:
	"""Convert RGB image or tensor image data to YCbCr format.
	Args:
		image: RGB image data read by ``PIL.Image''.
	Returns:
		YCbCr image array data.
	"""
	if type(image) == np.ndarray:
		y = 16. + (64.738 * image[:, :, 0] + 129.057 * image[:, :, 1] + 25.064 * image[:, :, 2]) / 256.
		cb = 128. + (-37.945 * image[:, :, 0] - 74.494 * image[:, :, 1] + 112.439 * image[:, :, 2]) / 256.
		cr = 128. + (112.439 * image[:, :, 0] - 94.154 * image[:, :, 1] - 18.285 * image[:, :, 2]) / 256.
		return np.array([y, cb, cr]).transpose([1, 2, 0])
	elif type(image) == torch.Tensor:
		if len(image.shape) == 4:
			image = image.squeeze(0)
		y = 16. + (64.738 * image[0, :, :] + 129.057 * image[1, :, :] + 25.064 * image[2, :, :]) / 256.
		cb = 128. + (-37.945 * image[0, :, :] - 74.494 * image[1, :, :] + 112.439 * image[2, :, :]) / 256.
		cr = 128. + (112.439 * image[0, :, :] - 94.154 * image[1, :, :] - 18.285 * image[2, :, :]) / 256.
		return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
	else:
		raise Exception("Unknown Type", type(image))


def convert_ycbcr_to_rgb(image: Any) -> Any:
	"""Convert YCbCr format image to RGB format.
	Args:
	   image: YCbCr image data read by ``PIL.Image''.
	Returns:
		RGB image array data.
	"""
	if type(image) == np.ndarray:
		r = 298.082 * image[:, :, 0] / 256. + 408.583 * image[:, :, 2] / 256. - 222.921
		g = 298.082 * image[:, :, 0] / 256. - 100.291 * image[:, :, 1] / 256. - 208.120 * image[:, :, 2] / 256. + 135.576
		b = 298.082 * image[:, :, 0] / 256. + 516.412 * image[:, :, 1] / 256. - 276.836
		return np.array([r, g, b]).transpose([1, 2, 0])
	elif type(image) == torch.Tensor:
		if len(image.shape) == 4:
			image = image.squeeze(0)
		r = 298.082 * image[0, :, :] / 256. + 408.583 * image[2, :, :] / 256. - 222.921
		g = 298.082 * image[0, :, :] / 256. - 100.291 * image[1, :, :] / 256. - 208.120 * image[2, :, :] / 256. + 135.576
		b = 298.082 * image[0, :, :] / 256. + 516.412 * image[1, :, :] / 256. - 276.836
		return torch.cat([r, g, b], 0).permute(1, 2, 0)
	else:
		raise Exception("Unknown Type", type(image))


def center_crop(lr: Any, hr: Any, image_size: int, upscale_factor: int) -> [Any, Any]:
	"""Cut ``PIL.Image`` in the center area of the image.
	Args:
		lr: Low-resolution image data read by ``PIL.Image``.
		hr: High-resolution image data read by ``PIL.Image``.
		image_size (int): The size of the captured image area. It should be the size of the high-resolution image.
		upscale_factor (int): magnification factor.
	Returns:
		Randomly cropped low-resolution images and high-resolution images.
	"""
	w, h = hr.size

	left = (w - image_size) // 2
	top = (h - image_size) // 2
	right = left + image_size
	bottom = top + image_size

	lr = lr.crop((left // upscale_factor,
				  top // upscale_factor,
				  right // upscale_factor,
				  bottom // upscale_factor))
	hr = hr.crop((left, top, right, bottom))

	return lr, hr


def random_crop(lr: Any, hr: Any, image_size: int, upscale_factor: int) -> [Any, Any]:
	"""Will ``PIL.Image`` randomly capture the specified area of the image.
	Args:
		lr: Low-resolution image data read by ``PIL.Image``.
		hr: High-resolution image data read by ``PIL.Image``.
		image_size (int): The size of the captured image area. It should be the size of the high-resolution image.
		upscale_factor (int): magnification factor.
	Returns:
		Randomly cropped low-resolution images and high-resolution images.
	"""
	w, h = hr.size
	left = torch.randint(0, w - image_size + 1, size=(1,)).item()
	top = torch.randint(0, h - image_size + 1, size=(1,)).item()
	right = left + image_size
	bottom = top + image_size

	lr = lr.crop((left // upscale_factor,
				  top // upscale_factor,
				  right // upscale_factor,
				  bottom // upscale_factor))
	hr = hr.crop((left, top, right, bottom))

	return lr, hr


def random_rotate(lr: Any, hr: Any, angle: int) -> [Any, Any]:
	"""Will ``PIL.Image`` randomly rotate the image.
	Args:
		lr: Low-resolution image data read by ``PIL.Image``.
		hr: High-resolution image data read by ``PIL.Image``.
		angle (int): rotation angle, clockwise and counterclockwise rotation.
	Returns:
		Randomly rotated low-resolution images and high-resolution images.
	"""
	angle = random.choice((+angle, -angle))
	lr = F.rotate(lr, angle)
	hr = F.rotate(hr, angle)

	return lr, hr


def random_horizontally_flip(lr: Any, hr: Any, p=0.5) -> [Any, Any]:
	"""Flip the ``PIL.Image`` image horizontally randomly.
	Args:
		lr: Low-resolution image data read by ``PIL.Image``.
		hr: High-resolution image data read by ``PIL.Image``.
		p (optional, float): rollover probability. (Default: 0.5)
	Returns:
		Low-resolution image and high-resolution image after random horizontal flip.
	"""
	if torch.rand(1).item() > p:
		lr = F.hflip(lr)
		hr = F.hflip(hr)

	return lr, hr


def random_vertically_flip(lr: Any, hr: Any, p=0.5) -> [Any, Any]:
	"""Turn the ``PIL.Image`` image upside down randomly.
	Args:
		lr: Low-resolution image data read by ``PIL.Image``.
		hr: High-resolution image data read by ``PIL.Image``.
		p (optional, float): rollover probability. (Default: 0.5)
	Returns:
		Randomly rotated up and down low-resolution images and high-resolution images.
	"""
	if torch.rand(1).item() > p:
		lr = F.vflip(lr)
		hr = F.vflip(hr)

	return lr, hr


def random_adjust_brightness(lr: Any, hr: Any) -> [Any, Any]:
	"""Set ``PIL.Image`` to randomly adjust the image brightness.
	Args:
		lr: Low-resolution image data read by ``PIL.Image``.
		hr: High-resolution image data read by ``PIL.Image``.
	Returns:
		Low-resolution image and high-resolution image with randomly adjusted brightness.
	"""
	# Randomly adjust the brightness gain range.
	factor = random.uniform(0.5, 2)
	lr = F.adjust_brightness(lr, factor)
	hr = F.adjust_brightness(hr, factor)

	return lr, hr


def random_adjust_contrast(lr: Any, hr: Any) -> [Any, Any]:
	"""Set ``PIL.Image`` to randomly adjust the image contrast.
	Args:
		lr: Low-resolution image data read by ``PIL.Image``.
		hr: High-resolution image data read by ``PIL.Image``.
	Returns:
		Low-resolution image and high-resolution image with randomly adjusted contrast.
	"""
	# Randomly adjust the contrast gain range.
	factor = random.uniform(0.5, 2)
	lr = F.adjust_contrast(lr, factor)
	hr = F.adjust_contrast(hr, factor)

	return lr, hr

#### metrics to compute -- assumes single images, i.e., tensor of 3 dims
def img_mae(x1, x2):
	m = torch.abs(x1-x2).mean()
	return m

def img_mse(x1, x2):
	m = torch.pow(torch.abs(x1-x2),2).mean()
	return m

def img_psnr(x1, x2):
	m = kornia.metrics.psnr(x1, x2, 1)
	return m

def img_ssim(x1, x2):
	m = kornia.metrics.ssim(x1.unsqueeze(0), x2.unsqueeze(0), 5)
	m = m.mean()
	return m

def show_SR_w_uncer(xLR, xHR, xSR, xSRvar, elim=(0,0.01), ulim=(0,0.15)):
	'''
	xLR/SR/HR: 3xHxW
	xSRvar: 1xHxW
	'''
	plt.figure(figsize=(30,10))
	
	plt.subplot(1,5,1)
	plt.imshow(xLR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1))
	plt.axis('off')
	
	plt.subplot(1,5,2)
	plt.imshow(xHR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1))
	plt.axis('off')
	
	plt.subplot(1,5,3)
	plt.imshow(xSR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1))
	plt.axis('off')
	
	plt.subplot(1,5,4)
	error_map = torch.mean(torch.pow(torch.abs(xSR-xHR),2), dim=0).to('cpu').data.unsqueeze(0) 
	print('error', error_map.min(), error_map.max())
	plt.imshow(error_map.transpose(0,2).transpose(0,1), cmap='jet')
	plt.clim(elim[0], elim[1])
	plt.axis('off')

	plt.subplot(1,5,5)
	print('uncer', xSRvar.min(), xSRvar.max())
	plt.imshow(xSRvar.to('cpu').data.transpose(0,2).transpose(0,1), cmap='hot')
	plt.clim(ulim[0], ulim[1])
	plt.axis('off')

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.show()

def show_SR_w_err(xLR, xHR, xSR, elim=(0,0.01), task=None, xMask=None):
	'''
	xLR/SR/HR: 3xHxW
	'''
	plt.figure(figsize=(30,10))
	
	if task != 'm':
		plt.subplot(1,4,1)
		plt.imshow(xLR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1))
		plt.axis('off')
		
		plt.subplot(1,4,2)
		plt.imshow(xHR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1))
		plt.axis('off')
		
		plt.subplot(1,4,3)
		plt.imshow(xSR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1))
		plt.axis('off')
	else:
		plt.subplot(1,4,1)
		plt.imshow(xLR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1), cmap='gray')
		plt.clim(0,0.9)
		plt.axis('off')
		
		plt.subplot(1,4,2)
		plt.imshow(xHR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1), cmap='gray')
		plt.clim(0,0.9)
		plt.axis('off')
		
		plt.subplot(1,4,3)
		plt.imshow(xSR.to('cpu').data.clip(0,1).transpose(0,2).transpose(0,1), cmap='gray')
		plt.clim(0,0.9)
		plt.axis('off')
	
	plt.subplot(1,4,4)
	if task == 'inpainting':
		error_map = torch.mean(torch.pow(torch.abs(xSR-xHR),2), dim=0).to('cpu').data.unsqueeze(0)*xMask.to('cpu').data
	else:
		error_map = torch.mean(torch.pow(torch.abs(xSR-xHR),2), dim=0).to('cpu').data.unsqueeze(0) 
	print('error', error_map.min(), error_map.max())
	plt.imshow(error_map.transpose(0,2).transpose(0,1), cmap='jet')
	plt.clim(elim[0], elim[1])
	plt.axis('off')

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.show()

def show_uncer4(xSRvar1, xSRvar2, xSRvar3, xSRvar4, ulim=(0,0.15)):
	'''
	xSRvar: 1xHxW
	'''
	plt.figure(figsize=(30,10))
	
	plt.subplot(1,4,1)
	print('uncer', xSRvar1.min(), xSRvar1.max())
	plt.imshow(xSRvar1.to('cpu').data.transpose(0,2).transpose(0,1), cmap='hot')
	plt.clim(ulim[0], ulim[1])
	plt.axis('off')

	plt.subplot(1,4,2)
	print('uncer', xSRvar2.min(), xSRvar2.max())
	plt.imshow(xSRvar2.to('cpu').data.transpose(0,2).transpose(0,1), cmap='hot')
	plt.clim(ulim[0], ulim[1])
	plt.axis('off')

	plt.subplot(1,4,3)
	print('uncer', xSRvar3.min(), xSRvar3.max())
	plt.imshow(xSRvar3.to('cpu').data.transpose(0,2).transpose(0,1), cmap='hot')
	plt.clim(ulim[0], ulim[1])
	plt.axis('off')

	plt.subplot(1,4,4)
	print('uncer', xSRvar4.min(), xSRvar4.max())
	plt.imshow(xSRvar4.to('cpu').data.transpose(0,2).transpose(0,1), cmap='hot')
	plt.clim(ulim[0], ulim[1])
	plt.axis('off')

	plt.subplots_adjust(wspace=0, hspace=0)
	plt.show()

##################### training BayesCap
def train_BayesCap(
	NetC,
	NetG,
	train_loader,
	eval_loader,
	Cri = TempCombLoss(),
	device='cuda',
	dtype=torch.cuda.FloatTensor(),
	init_lr=1e-4,
	num_epochs=100,
	eval_every=1,
	ckpt_path='../ckpt/BayesCap',
	T1=1e0,
	T2=5e-2,
	task=None,
):
	NetC.to(device)
	NetC.train()
	NetG.to(device)
	NetG.eval()
	optimizer = torch.optim.Adam(list(NetC.parameters()), lr=init_lr)
	optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

	score = -1e8
	all_loss = []
	for eph in range(num_epochs):
		eph_loss = 0
		with tqdm(train_loader, unit='batch') as tepoch:
			for (idx, batch) in enumerate(tepoch):
				if idx>2000:
					break
				tepoch.set_description('Epoch {}'.format(eph))
				##
				xLR, xHR = batch[0].to(device), batch[1].to(device)
				xLR, xHR = xLR.type(dtype), xHR.type(dtype)
				if task == 'inpainting':
					xMask = random_mask(xLR.shape[0], (xLR.shape[2], xLR.shape[3]))
					xMask = xMask.to(device).type(dtype)
				# pass them through the network
				with torch.no_grad():
					if task == 'inpainting':
						_, xSR1 = NetG(xLR, xMask)
					elif task == 'depth':
						xSR1 = NetG(xLR)[("disp", 0)]
					else:
						xSR1 = NetG(xLR)
				# with torch.autograd.set_detect_anomaly(True):
				xSR = xSR1.clone()
				xSRC_mu, xSRC_alpha, xSRC_beta = NetC(xSR)
				# print(xSRC_alpha) 
				optimizer.zero_grad()
				loss = Cri(xSRC_mu, xSRC_alpha, xSRC_beta, xSR, xHR, T1=T1,T2=T2)
				# print(loss)
				loss.backward()
				optimizer.step()
				##
				eph_loss += loss.item()
				tepoch.set_postfix(loss=loss.item())
			eph_loss /= len(train_loader)
			all_loss.append(eph_loss)
			print('Avg. loss: {}'.format(eph_loss))
		# evaluate and save the models
		torch.save(NetC.state_dict(), ckpt_path+'_last.pth')
		if eph%eval_every == 0:
			curr_score = eval_BayesCap(
				NetC,
				NetG,
				eval_loader,
				device=device,
				dtype=dtype,
				task=task,
			)
			print('current score: {} | Last best score: {}'.format(curr_score, score))
			if curr_score >= score:
				score = curr_score
				torch.save(NetC.state_dict(), ckpt_path+'_best.pth')
	optim_scheduler.step()

#### get different uncertainty maps
def get_uncer_BayesCap(
	NetC,
	NetG,
	xin,
	task=None,
	xMask=None,
):
	with torch.no_grad():
		if task == 'inpainting':
			_, xSR = NetG(xin, xMask)
		else:
			xSR = NetG(xin)
		xSRC_mu, xSRC_alpha, xSRC_beta = NetC(xSR)
	a_map = (1/(xSRC_alpha + 1e-5)).to('cpu').data
	b_map = xSRC_beta.to('cpu').data
	xSRvar = (a_map**2)*(torch.exp(torch.lgamma(3/(b_map + 1e-2)))/torch.exp(torch.lgamma(1/(b_map + 1e-2))))

	return xSRvar

def get_uncer_TTDAp(
	NetG,
	xin,
	p_mag=0.05,
	num_runs=50,
	task=None,
	xMask=None,
):
	list_xSR = []
	with torch.no_grad():
		for z in range(num_runs):
			if task == 'inpainting':
				_, xSRz = NetG(xin+p_mag*xin.max()*torch.randn_like(xin), xMask)
			else:
				xSRz = NetG(xin+p_mag*xin.max()*torch.randn_like(xin))
			list_xSR.append(xSRz)
	xSRmean = torch.mean(torch.cat(list_xSR, dim=0), dim=0).unsqueeze(0)
	xSRvar = torch.mean(torch.var(torch.cat(list_xSR, dim=0), dim=0), dim=0).unsqueeze(0).unsqueeze(1)
	return xSRvar

def get_uncer_DO(
	NetG,
	xin,
	dop=0.2,
	num_runs=50,
	task=None,
	xMask=None,
):
	list_xSR = []
	with torch.no_grad():
		for z in range(num_runs):
			if task == 'inpainting':
				_, xSRz = NetG(xin, xMask, dop=dop)
			else:
				xSRz = NetG(xin, dop=dop)
			list_xSR.append(xSRz)
	xSRmean = torch.mean(torch.cat(list_xSR, dim=0), dim=0).unsqueeze(0)
	xSRvar = torch.mean(torch.var(torch.cat(list_xSR, dim=0), dim=0), dim=0).unsqueeze(0).unsqueeze(1)
	return xSRvar
