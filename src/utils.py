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

def get_UCE(list_err, list_yout_var, num_bins=100):
	err_min = np.min(list_err)
	err_max = np.max(list_err)
	err_len = (err_max-err_min)/num_bins
	num_points = len(list_err)
	
	bin_stats = {}
	for i in range(num_bins):
		bin_stats[i] = {
			'start_idx': err_min + i*err_len,
			'end_idx': err_min + (i+1)*err_len,
			'num_points': 0,
			'mean_err': 0,
			'mean_var': 0,
		}
	
	for e,v in zip(list_err, list_yout_var):
		for i in range(num_bins):
			if e>=bin_stats[i]['start_idx'] and e<bin_stats[i]['end_idx']:
				bin_stats[i]['num_points'] += 1
				bin_stats[i]['mean_err'] += e
				bin_stats[i]['mean_var'] += v
	
	uce = 0
	eps = 1e-8
	for i in range(num_bins):
		bin_stats[i]['mean_err'] /= bin_stats[i]['num_points'] + eps
		bin_stats[i]['mean_var'] /= bin_stats[i]['num_points'] + eps
		bin_stats[i]['uce_bin'] = (bin_stats[i]['num_points']/num_points) \
			*(np.abs(bin_stats[i]['mean_err'] - bin_stats[i]['mean_var']))
		uce += bin_stats[i]['uce_bin']
	
	list_x, list_y = [], []
	for i in range(num_bins):
		if bin_stats[i]['num_points']>0:
			list_x.append(bin_stats[i]['mean_err'])
			list_y.append(bin_stats[i]['mean_var'])
	
	# sns.set_style('darkgrid')
	# sns.scatterplot(x=list_x, y=list_y)
	# sns.regplot(x=list_x, y=list_y, order=1)
	# plt.xlabel('MSE', fontsize=34)
	# plt.ylabel('Uncertainty', fontsize=34)
	# plt.plot(list_x, list_x, color='r')
	# plt.xlim(np.min(list_x), np.max(list_x))
	# plt.ylim(np.min(list_err), np.max(list_x))
	# plt.show()

	return bin_stats, uce

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
				if task == 'depth':
					loss = Cri(xSRC_mu, xSRC_alpha, xSRC_beta, xSR, T1=T1, T2=T2)
				else:
					loss = Cri(xSRC_mu, xSRC_alpha, xSRC_beta, xHR, T1=T1, T2=T2)
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

################### Different eval functions

def eval_BayesCap(
	NetC,
	NetG,
	eval_loader,
	device='cuda',
	dtype=torch.cuda.FloatTensor,
	task=None,
	xMask=None,
):
	NetC.to(device)
	NetC.eval()
	NetG.to(device)
	NetG.eval()
	
	mean_ssim = 0
	mean_psnr = 0
	mean_mse = 0
	mean_mae = 0
	num_imgs = 0
	list_error = []
	list_var = []
	with tqdm(eval_loader, unit='batch') as tepoch:
		for (idx, batch) in enumerate(tepoch):
			tepoch.set_description('Validating ...')
			##
			xLR, xHR = batch[0].to(device), batch[1].to(device)
			xLR, xHR = xLR.type(dtype), xHR.type(dtype)
			if task == 'inpainting':
				if xMask==None:
					xMask = random_mask(xLR.shape[0], (xLR.shape[2], xLR.shape[3]))
					xMask = xMask.to(device).type(dtype)
				else:
					xMask = xMask.to(device).type(dtype)
			# pass them through the network
			with torch.no_grad():
				if task == 'inpainting':
					_, xSR = NetG(xLR, xMask)
				elif task == 'depth':
					xSR = NetG(xLR)[("disp", 0)]
				else:
					xSR = NetG(xLR)
				xSRC_mu, xSRC_alpha, xSRC_beta = NetC(xSR)
			a_map = (1/(xSRC_alpha + 1e-5)).to('cpu').data
			b_map = xSRC_beta.to('cpu').data
			xSRvar = (a_map**2)*(torch.exp(torch.lgamma(3/(b_map + 1e-2)))/torch.exp(torch.lgamma(1/(b_map + 1e-2))))
			n_batch = xSRC_mu.shape[0]
			if task == 'depth':
				xHR = xSR
			for j in range(n_batch):
				num_imgs += 1
				mean_ssim += img_ssim(xSRC_mu[j], xHR[j])
				mean_psnr += img_psnr(xSRC_mu[j], xHR[j])
				mean_mse += img_mse(xSRC_mu[j], xHR[j])
				mean_mae += img_mae(xSRC_mu[j], xHR[j])

				show_SR_w_uncer(xLR[j], xHR[j], xSR[j], xSRvar[j])
				
				error_map = torch.mean(torch.pow(torch.abs(xSR[j]-xHR[j]),2), dim=0).to('cpu').data.reshape(-1)
				var_map =  xSRvar[j].to('cpu').data.reshape(-1)
				list_error.extend(list(error_map.numpy()))
				list_var.extend(list(var_map.numpy()))
			##
		mean_ssim /= num_imgs
		mean_psnr /= num_imgs
		mean_mse /= num_imgs
		mean_mae /= num_imgs
		print(
			'Avg. SSIM: {} | Avg. PSNR: {} | Avg. MSE: {} | Avg. MAE: {}'.format
			(
				mean_ssim, mean_psnr, mean_mse, mean_mae 
			)
		)
		# print(len(list_error), len(list_var))
		# print('UCE: ', get_UCE(list_error[::10], list_var[::10], num_bins=500)[1])
		# print('C.Coeff: ', np.corrcoef(np.array(list_error[::10]), np.array(list_var[::10])))
	return mean_ssim

def eval_TTDA_p(
	NetG,
	eval_loader,
	device='cuda',
	dtype=torch.cuda.FloatTensor,
	p_mag=0.05,
	num_runs=50,
	task = None,
	xMask = None,
):
	NetG.to(device)
	NetG.eval()
	
	mean_ssim = 0
	mean_psnr = 0
	mean_mse = 0
	mean_mae = 0
	num_imgs = 0
	with tqdm(eval_loader, unit='batch') as tepoch:
		for (idx, batch) in enumerate(tepoch):
			tepoch.set_description('Validating ...')
			##
			xLR, xHR = batch[0].to(device), batch[1].to(device)
			xLR, xHR = xLR.type(dtype), xHR.type(dtype)
			# pass them through the network
			list_xSR = []
			with torch.no_grad():
				if task=='inpainting':
					_, xSR = NetG(xLR, xMask)
				else:
					xSR = NetG(xLR)
				for z in range(num_runs):
					xSRz = NetG(xLR+p_mag*xLR.max()*torch.randn_like(xLR))	
					list_xSR.append(xSRz)
			xSRmean = torch.mean(torch.cat(list_xSR, dim=0), dim=0).unsqueeze(0)
			xSRvar = torch.mean(torch.var(torch.cat(list_xSR, dim=0), dim=0), dim=0).unsqueeze(0).unsqueeze(1)
			n_batch = xSR.shape[0]
			for j in range(n_batch):
				num_imgs += 1
				mean_ssim += img_ssim(xSR[j], xHR[j])
				mean_psnr += img_psnr(xSR[j], xHR[j])
				mean_mse += img_mse(xSR[j], xHR[j])
				mean_mae += img_mae(xSR[j], xHR[j])

				show_SR_w_uncer(xLR[j], xHR[j], xSR[j], xSRvar[j])
				
		mean_ssim /= num_imgs
		mean_psnr /= num_imgs
		mean_mse /= num_imgs
		mean_mae /= num_imgs
		print(
			'Avg. SSIM: {} | Avg. PSNR: {} | Avg. MSE: {} | Avg. MAE: {}'.format
			(
				mean_ssim, mean_psnr, mean_mse, mean_mae 
			)
		)
		
	return mean_ssim

def eval_DO(
	NetG,
	eval_loader,
	device='cuda',
	dtype=torch.cuda.FloatTensor,
	dop=0.2,
	num_runs=50,
	task=None,
	xMask=None,
):
	NetG.to(device)
	NetG.eval()
	
	mean_ssim = 0
	mean_psnr = 0
	mean_mse = 0
	mean_mae = 0
	num_imgs = 0
	with tqdm(eval_loader, unit='batch') as tepoch:
		for (idx, batch) in enumerate(tepoch):
			tepoch.set_description('Validating ...')
			##
			xLR, xHR = batch[0].to(device), batch[1].to(device)
			xLR, xHR = xLR.type(dtype), xHR.type(dtype)
			# pass them through the network
			list_xSR = []
			with torch.no_grad():
				if task == 'inpainting':
					_, xSR = NetG(xLR, xMask)
				else:
					xSR = NetG(xLR)
				for z in range(num_runs):
					xSRz = NetG(xLR, dop=dop)	
					list_xSR.append(xSRz)
			xSRmean = torch.mean(torch.cat(list_xSR, dim=0), dim=0).unsqueeze(0)
			xSRvar = torch.mean(torch.var(torch.cat(list_xSR, dim=0), dim=0), dim=0).unsqueeze(0).unsqueeze(1)
			n_batch = xSR.shape[0]
			for j in range(n_batch):
				num_imgs += 1
				mean_ssim += img_ssim(xSR[j], xHR[j])
				mean_psnr += img_psnr(xSR[j], xHR[j])
				mean_mse += img_mse(xSR[j], xHR[j])
				mean_mae += img_mae(xSR[j], xHR[j])

				show_SR_w_uncer(xLR[j], xHR[j], xSR[j], xSRvar[j])
						##
		mean_ssim /= num_imgs
		mean_psnr /= num_imgs
		mean_mse /= num_imgs
		mean_mae /= num_imgs
		print(
			'Avg. SSIM: {} | Avg. PSNR: {} | Avg. MSE: {} | Avg. MAE: {}'.format
			(
				mean_ssim, mean_psnr, mean_mse, mean_mae 
			)
		)
	
	return mean_ssim


################# Degrading Identity 
def degrage_BayesCap_p(
	NetC,
	NetG,
	eval_loader,
	device='cuda',
	dtype=torch.cuda.FloatTensor,
	num_runs=50,
):
	NetC.to(device)
	NetC.eval()
	NetG.to(device)
	NetG.eval()
	
	p_mag_list = [0, 0.05, 0.1, 0.15, 0.2]
	list_s = []
	list_p = []
	list_u1 = []
	list_u2 = []
	list_c = []
	for p_mag in p_mag_list:
		mean_ssim = 0
		mean_psnr = 0
		mean_mse = 0
		mean_mae = 0
		num_imgs = 0
		list_error = []
		list_error2 = []
		list_var = []

		with tqdm(eval_loader, unit='batch') as tepoch:
			for (idx, batch) in enumerate(tepoch):
				tepoch.set_description('Validating ...')
				##
				xLR, xHR = batch[0].to(device), batch[1].to(device)
				xLR, xHR = xLR.type(dtype), xHR.type(dtype)
				# pass them through the network
				with torch.no_grad():
					xSR = NetG(xLR)
					xSRC_mu, xSRC_alpha, xSRC_beta = NetC(xSR + p_mag*xSR.max()*torch.randn_like(xSR))
				a_map = (1/(xSRC_alpha + 1e-5)).to('cpu').data
				b_map = xSRC_beta.to('cpu').data
				xSRvar = (a_map**2)*(torch.exp(torch.lgamma(3/(b_map + 1e-2)))/torch.exp(torch.lgamma(1/(b_map + 1e-2))))
				n_batch = xSRC_mu.shape[0]
				for j in range(n_batch):
					num_imgs += 1
					mean_ssim += img_ssim(xSRC_mu[j], xSR[j])
					mean_psnr += img_psnr(xSRC_mu[j], xSR[j])
					mean_mse += img_mse(xSRC_mu[j], xSR[j])
					mean_mae += img_mae(xSRC_mu[j], xSR[j])

					error_map = torch.mean(torch.pow(torch.abs(xSR[j]-xHR[j]),2), dim=0).to('cpu').data.reshape(-1)
					error_map2 = torch.mean(torch.pow(torch.abs(xSRC_mu[j]-xHR[j]),2), dim=0).to('cpu').data.reshape(-1)
					var_map =  xSRvar[j].to('cpu').data.reshape(-1)
					list_error.extend(list(error_map.numpy()))
					list_error2.extend(list(error_map2.numpy()))
					list_var.extend(list(var_map.numpy()))
				##
			mean_ssim /= num_imgs
			mean_psnr /= num_imgs
			mean_mse /= num_imgs
			mean_mae /= num_imgs
			print(
				'Avg. SSIM: {} | Avg. PSNR: {} | Avg. MSE: {} | Avg. MAE: {}'.format
				(
					mean_ssim, mean_psnr, mean_mse, mean_mae 
				)
			)
			uce1 = get_UCE(list_error[::100], list_var[::100], num_bins=200)[1]
			uce2 = get_UCE(list_error2[::100], list_var[::100], num_bins=200)[1]
			print('UCE1: ', uce1)
			print('UCE2: ', uce2)
			list_s.append(mean_ssim.item())
			list_p.append(mean_psnr.item())
			list_u1.append(uce1)
			list_u2.append(uce2)
	
	plt.plot(list_s)
	plt.show()
	plt.plot(list_p)
	plt.show()

	plt.plot(list_u1, label='wrt SR output')
	plt.plot(list_u2, label='wrt BayesCap output')
	plt.legend()
	plt.show()

	sns.set_style('darkgrid')
	fig,ax = plt.subplots()
	# make a plot
	ax.plot(p_mag_list, list_s, color="red", marker="o")
	# set x-axis label
	ax.set_xlabel("Reducing faithfulness of BayesCap Reconstruction",fontsize=10)
	# set y-axis label
	ax.set_ylabel("SSIM btwn BayesCap and SRGAN outputs", color="red",fontsize=10)

	# twin object for two different y-axis on the sample plot
	ax2=ax.twinx()
	# make a plot with different y-axis using second axis object
	ax2.plot(p_mag_list, list_u1, color="blue", marker="o", label='UCE wrt to error btwn SRGAN output and GT')
	ax2.plot(p_mag_list, list_u2, color="orange", marker="o", label='UCE wrt to error btwn BayesCap output and GT')
	ax2.set_ylabel("UCE", color="green", fontsize=10)
	plt.legend(fontsize=10)
	plt.tight_layout()
	plt.show()

################# DeepFill_v2
   
# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_names(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.jpg'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim

## for contextual attention

def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x

def random_mask(num_batch=1, mask_shape=(256,256)):
    list_mask = []
    for _ in range(num_batch):
        # rectangle mask
        image_height = mask_shape[0]
        image_width = mask_shape[1]
        max_delta_height = image_height//8
        max_delta_width = image_width//8
        height = image_height//4
        width = image_width//4
        max_t = image_height - height
        max_l = image_width - width
        t = random.randint(0, max_t)
        l = random.randint(0, max_l)
        # bbox = (t, l, height, width)
        h = random.randint(0, max_delta_height//2)
        w = random.randint(0, max_delta_width//2)
        mask = torch.zeros((1, 1, image_height, image_width))
        mask[:, :, t+h:t+height-h, l+w:l+width-w] = 1
        rect_mask = mask

        # brush mask
        min_num_vertex = 4
        max_num_vertex = 12
        mean_angle = 2 * math.pi / 5
        angle_range = 2 * math.pi / 15
        min_width = 12
        max_width = 40
        H, W = image_height, image_width
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('L', (W, H), 0)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=255, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=255)

        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)

        mask = transforms.ToTensor()(mask)
        mask = mask.reshape((1, 1, H, W))
        brush_mask = mask

        mask = torch.cat([rect_mask, brush_mask], dim=1).max(dim=1, keepdim=True)[0]
        list_mask.append(mask)
    mask = torch.cat(list_mask, dim=0)
    return mask