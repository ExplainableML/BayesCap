import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

class ContentLoss(nn.Module):
	"""Constructs a content loss function based on the VGG19 network.
	Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

	Paper reference list:
		-`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
		-`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
		-`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

	 """

	def __init__(self) -> None:
		super(ContentLoss, self).__init__()
		# Load the VGG19 model trained on the ImageNet dataset.
		vgg19 = models.vgg19(pretrained=True).eval()
		# Extract the thirty-sixth layer output in the VGG19 model as the content loss.
		self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])
		# Freeze model parameters.
		for parameters in self.feature_extractor.parameters():
			parameters.requires_grad = False

		# The preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset.
		self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

	def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
		# Standardized operations
		sr = sr.sub(self.mean).div(self.std)
		hr = hr.sub(self.mean).div(self.std)

		# Find the feature map difference between the two images
		loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

		return loss


class GenGaussLoss(nn.Module):
	def __init__(
		self, reduction='mean',
		alpha_eps = 1e-4, beta_eps=1e-4,
		resi_min = 1e-4, resi_max=1e3
	) -> None:
		super(GenGaussLoss, self).__init__()
		self.reduction = reduction
		self.alpha_eps = alpha_eps
		self.beta_eps = beta_eps
		self.resi_min = resi_min
		self.resi_max = resi_max
	
	def forward(
		self, 
		mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target: Tensor
	):
		one_over_alpha1 = one_over_alpha + self.alpha_eps
		beta1 = beta + self.beta_eps

		resi = torch.abs(mean - target)
		# resi = torch.pow(resi*one_over_alpha1, beta1).clamp(min=self.resi_min, max=self.resi_max)
		resi = (resi*one_over_alpha1*beta1).clamp(min=self.resi_min, max=self.resi_max)
		## check if resi has nans
		if torch.sum(resi != resi) > 0:
			print('resi has nans!!')
			return None
		
		log_one_over_alpha = torch.log(one_over_alpha1)
		log_beta = torch.log(beta1)
		lgamma_beta = torch.lgamma(torch.pow(beta1, -1))
		
		if torch.sum(log_one_over_alpha != log_one_over_alpha) > 0:
			print('log_one_over_alpha has nan')
		if torch.sum(lgamma_beta != lgamma_beta) > 0:
			print('lgamma_beta has nan')
		if torch.sum(log_beta != log_beta) > 0:
			print('log_beta has nan')
		
		l = resi - log_one_over_alpha + lgamma_beta - log_beta

		if self.reduction == 'mean':
			return l.mean()
		elif self.reduction == 'sum':
			return l.sum()
		else:
			print('Reduction not supported')
			return None

class TempCombLoss(nn.Module):
	def __init__(
		self, reduction='mean',
		alpha_eps = 1e-4, beta_eps=1e-4,
		resi_min = 1e-4, resi_max=1e3
	) -> None:
		super(TempCombLoss, self).__init__()
		self.reduction = reduction
		self.alpha_eps = alpha_eps
		self.beta_eps = beta_eps
		self.resi_min = resi_min
		self.resi_max = resi_max

		self.L_GenGauss = GenGaussLoss(
			reduction=self.reduction,
			alpha_eps=self.alpha_eps, beta_eps=self.beta_eps, 
			resi_min=self.resi_min, resi_max=self.resi_max
		)
		self.L_l1 = nn.L1Loss(reduction=self.reduction)
	
	def forward(
		self,
		mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target1: Tensor, target2: Tensor,
		T1: float, T2: float
	):
		##target1 is the base model output for identity mapping
		##target2 is the ground truth for the GenGauss loss
		l1 = self.L_l1(mean, target1)
		l2 = self.L_GenGauss(mean, one_over_alpha, beta, target2)
		l = T1*l1 + T2*l2

		return l


# x1 = torch.randn(4,3,32,32)
# x2 = torch.rand(4,3,32,32)
# x3 = torch.rand(4,3,32,32)
# x4 = torch.randn(4,3,32,32)

# L = GenGaussLoss(alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e3)
# L2 =  TempCombLoss(alpha_eps=1e-4, beta_eps=1e-4, resi_min=1e-4, resi_max=1e3)
# print(L(x1, x2, x3, x4), L2(x1, x2, x3, x4, 1e0, 1e-2))