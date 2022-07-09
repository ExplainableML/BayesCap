import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

# __all__ = [
#     "ResidualConvBlock",
#     "Discriminator", "Generator",
# ]


class ResidualConvBlock(nn.Module):
	"""Implements residual conv function.

	Args:
		channels (int): Number of channels in the input image.
	"""

	def __init__(self, channels: int) -> None:
		super(ResidualConvBlock, self).__init__()
		self.rcb = nn.Sequential(
			nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(channels),
			nn.PReLU(),
			nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(channels),
		)

	def forward(self, x: Tensor) -> Tensor:
		identity = x

		out = self.rcb(x)
		out = torch.add(out, identity)

		return out


class Discriminator(nn.Module):
	def __init__(self) -> None:
		super(Discriminator, self).__init__()
		self.features = nn.Sequential(
			# input size. (3) x 96 x 96
			nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=False),
			nn.LeakyReLU(0.2, True),
			# state size. (64) x 48 x 48
			nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, True),
			# state size. (128) x 24 x 24
			nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, True),
			# state size. (256) x 12 x 12
			nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, True),
			# state size. (512) x 6 x 6
			nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, True),
		)

		self.classifier = nn.Sequential(
			nn.Linear(512 * 6 * 6, 1024),
			nn.LeakyReLU(0.2, True),
			nn.Linear(1024, 1),
		)

	def forward(self, x: Tensor) -> Tensor:
		out = self.features(x)
		out = torch.flatten(out, 1)
		out = self.classifier(out)

		return out


class Generator(nn.Module):
	def __init__(self) -> None:
		super(Generator, self).__init__()
		# First conv layer.
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(3, 64, (9, 9), (1, 1), (4, 4)),
			nn.PReLU(),
		)

		# Features trunk blocks.
		trunk = []
		for _ in range(16):
			trunk.append(ResidualConvBlock(64))
		self.trunk = nn.Sequential(*trunk)

		# Second conv layer.
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
			nn.BatchNorm2d(64),
		)

		# Upscale conv block.
		self.upsampling = nn.Sequential(
			nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
			nn.PixelShuffle(2),
			nn.PReLU(),
			nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
			nn.PixelShuffle(2),
			nn.PReLU(),
		)

		# Output layer.
		self.conv_block3 = nn.Conv2d(64, 3, (9, 9), (1, 1), (4, 4))

		# Initialize neural network weights.
		self._initialize_weights()

	def forward(self, x: Tensor, dop=None) -> Tensor:
		if not dop:
			return self._forward_impl(x)
		else:
			return self._forward_w_dop_impl(x, dop)

	# Support torch.script function.
	def _forward_impl(self, x: Tensor) -> Tensor:
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = self.conv_block2(out)
		out = torch.add(out1, out2)
		out = self.upsampling(out)
		out = self.conv_block3(out)

		return out
	
	def _forward_w_dop_impl(self, x: Tensor, dop) -> Tensor:
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = F.dropout2d(self.conv_block2(out), p=dop)
		out = torch.add(out1, out2)
		out = self.upsampling(out)
		out = self.conv_block3(out)

		return out

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)


#### BayesCap
class BayesCap(nn.Module):
	def __init__(self, in_channels=3, out_channels=3) -> None:
		super(BayesCap, self).__init__()
		# First conv layer.
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(
				in_channels, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
		)

		# Features trunk blocks.
		trunk = []
		for _ in range(16):
			trunk.append(ResidualConvBlock(64))
		self.trunk = nn.Sequential(*trunk)

		# Second conv layer.
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=3, stride=1, padding=1, bias=False
			),
			nn.BatchNorm2d(64),
		)

		# Output layer.
		self.conv_block3_mu = nn.Conv2d(
			64, out_channels=out_channels, 
			kernel_size=9, stride=1, padding=4
		)
		self.conv_block3_alpha = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)
		self.conv_block3_beta = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)

		# Initialize neural network weights.
		self._initialize_weights()

	def forward(self, x: Tensor) -> Tensor:
		return self._forward_impl(x)

	# Support torch.script function.
	def _forward_impl(self, x: Tensor) -> Tensor:
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = self.conv_block2(out)
		out = out1 + out2
		out_mu = self.conv_block3_mu(out)
		out_alpha = self.conv_block3_alpha(out)
		out_beta = self.conv_block3_beta(out)
		return out_mu, out_alpha, out_beta

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)
                
                
class BayesCap_noID(nn.Module):
	def __init__(self, in_channels=3, out_channels=3) -> None:
		super(BayesCap_noID, self).__init__()
		# First conv layer.
		self.conv_block1 = nn.Sequential(
			nn.Conv2d(
				in_channels, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
		)

		# Features trunk blocks.
		trunk = []
		for _ in range(16):
			trunk.append(ResidualConvBlock(64))
		self.trunk = nn.Sequential(*trunk)

		# Second conv layer.
		self.conv_block2 = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=3, stride=1, padding=1, bias=False
			),
			nn.BatchNorm2d(64),
		)

		# Output layer.
		# self.conv_block3_mu = nn.Conv2d(
		# 	64, out_channels=out_channels, 
		# 	kernel_size=9, stride=1, padding=4
		# )
		self.conv_block3_alpha = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)
		self.conv_block3_beta = nn.Sequential(
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 64, 
				kernel_size=9, stride=1, padding=4
			),
			nn.PReLU(),
			nn.Conv2d(
				64, 1, 
				kernel_size=9, stride=1, padding=4
			),
			nn.ReLU(),
		)

		# Initialize neural network weights.
		self._initialize_weights()

	def forward(self, x: Tensor) -> Tensor:
		return self._forward_impl(x)

	# Support torch.script function.
	def _forward_impl(self, x: Tensor) -> Tensor:
		out1 = self.conv_block1(x)
		out = self.trunk(out1)
		out2 = self.conv_block2(out)
		out = out1 + out2
		# out_mu = self.conv_block3_mu(out)
		out_alpha = self.conv_block3_alpha(out)
		out_beta = self.conv_block3_beta(out)
		return out_alpha, out_beta

	def _initialize_weights(self) -> None:
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.kaiming_normal_(module.weight)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)