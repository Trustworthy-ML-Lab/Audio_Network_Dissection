import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from BEATs.BEATs import BEATs, BEATsConfig


class MyBeatsModel(nn.Module):
	def __init__(self, checkpoint_path, num_class=50, freeze_ssl_backbone=True):
		super(MyBeatsModel, self).__init__()

		self.freeze_ssl_backbone = freeze_ssl_backbone

		# FIXME without saving cfg
		with open('BEATs/cfg.pkl', 'rb') as f:
			checkpoint_cfg = pickle.load(f)

		self.cfg = BEATsConfig(checkpoint_cfg)

		checkpoint = torch.load(checkpoint_path)
		self.beats = BEATs(self.cfg)
		if freeze_ssl_backbone:
			for param in self.beats.parameters():
				param.requires_grad = False

		self.fc = nn.Linear(self.cfg.encoder_embed_dim, num_class)

		# self.beats.load_state_dict(checkpoint)
		# for a in checkpoint.keys():
		# 	if "fc" in a: 
		# 		print(1, a)

		own_state = self.beats.state_dict()
		# for a in own_state.keys():
		# 	if "fc" in a: 
		# 		print(a)

		for name, param in checkpoint.items():
			if name == "fc.weight" or name == "fc.bias":
				self.fc.state_dict()[name.replace("fc.", "")].copy_(param)
			else: 
				param = param.data
				own_state[name.replace("beats.", "")].copy_(param)





	def forward(self, x, padding_mask=None):
		"""Forward pass. Return x"""

		# https://pytorch.org/audio/stable/transforms.html


		# Get the representation
		if padding_mask != None:
			x, _ = self.beats.extract_features(x, padding_mask)
		else:
			x, _ = self.beats.extract_features(x)

		# Get the logits
		x = self.fc(x)

		# Mean pool the second layer
		x = x.mean(dim=1)

		return x
	
	def get_optimizer(self, lr=1e-3):

		# if not self.freeze_ssl_backbone:
		optimizer = optim.AdamW(
		[{"params": self.beats.parameters()}, {"params": self.fc.parameters()}],
			lr=lr, betas=(0.9, 0.98), weight_decay=0.01
		)  
		# else:
		# 	optimizer = optim.AdamW(
		# 		self.fc.parameters(),
		# 		lr=lr, betas=(0.9, 0.98), weight_decay=0.01
		# 	)

		return optimizer
