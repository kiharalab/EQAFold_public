import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock2D, ResNet

from .alphafold2 import *
from .invariant_point_attention import *

class linear(nn.Module):
	def __init__(self, args):
		super(linear, self).__init__()

		kernel_size = 3
		padding = int((kernel_size - 1)/2)
		model = args.model

		out_channels_dist = args.out_channels_dist
		self.out_channels_angle = args.out_channels_angle
		out_channels_mu = args.out_channels_mu
		out_channels_theta = args.out_channels_theta
		out_channels_sce = args.out_channels_sce
		out_channels_no = args.out_channels_no
		self.use_templates = args.use_templates
		self.embed_type = args.embed

		self.model = model
		num_blocks = args.num_blocks
		dropout = args.dropout
		dilation_list = args.dilations
		channels = args.channels

		self.e2e = args.e2e
		self.msa = args.msa

		self.recycle = args.recycle
		self.device_id = args.device_id
		self.use_cuda = args.cuda
		evo_dim = 64
		if self.embed_type == 'msa_transformer':
			self.msa_project = nn.Sequential(nn.Linear(256, evo_dim))
			self.pairwise_project = nn.Sequential(nn.Linear(128, evo_dim))
		if model == 'resnet':
			self.resnet = ResNet(in_channels=128, channels=channels, num_blocks=num_blocks, dropout=dropout, dilation_list=dilation_list, kernel_size=kernel_size, padding=padding)
		
		if model == 'alphafold':	
			print("Using evoformer from alphafold2", num_blocks)
			#self.outer_mean = OuterMean(dim=evo_dim)
			self.resnet = Evoformer(
				dim = evo_dim,
				depth = num_blocks,
				seq_len = 1024,
				heads = 8,
				dim_head = 32,
				attn_dropout = 0,
				ff_dropout = 0.2
        	)
			self.single_project = nn.Sequential(nn.Linear(evo_dim, 384))
			if self.msa:
				self.msa_project2 = nn.Sequential(nn.Linear(evo_dim, 23))
			if self.recycle:
				self.point_project = nn.Sequential(nn.Linear(15, evo_dim))
				self.norm1 = nn.LayerNorm(evo_dim)
				self.norm2 = nn.LayerNorm(evo_dim)
			
			self.regression = nn.Sequential(
				nn.Linear(384, 128),
				nn.ReLU(),
				nn.Linear(128, 128),
				nn.ReLU(),
				nn.Linear(128, 3),
			)
			self.mse = nn.MSELoss()
		pairwise_repr_dim_default =64 if self.e2e else 128
		self.ipa = IPATransformer(dim=384, num_tokens = 21, depth=8, require_pairwise_repr = True, predict_points = True, pairwise_repr_dim_default=pairwise_repr_dim_default)
		if self.use_templates:
			#self.template_project = nn.Sequential(nn.Linear(64 * 5, 64))
			self.template_project = nn.Sequential(
				nn.Conv2d(5*64, 64, kernel_size=kernel_size, padding=padding),
				nn.InstanceNorm2d(64, affine=True),
				nn.ELU(),
				nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding),
				nn.InstanceNorm2d(64, affine=True),
				nn.ELU()
			)
		self.conv_dist = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
			nn.InstanceNorm2d(channels, affine=True),
			nn.ELU(),
			nn.Conv2d(channels, out_channels_dist, kernel_size=1, stride=1, padding=0)
		)
		
		self.conv_mu = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
			nn.InstanceNorm2d(channels, affine=True),
			nn.ELU(),
			nn.Conv2d(channels, out_channels_mu, kernel_size=1, stride=1, padding=0)
		)

		self.conv_theta = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
			nn.InstanceNorm2d(channels, affine=True),
			nn.ELU(),
			nn.Conv2d(channels, out_channels_theta, kernel_size=1, stride=1, padding=0)
		)

		self.conv_rho = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
			nn.InstanceNorm2d(channels, affine=True),
			nn.ELU(),
			nn.Conv2d(channels, out_channels_theta, kernel_size=1, stride=1, padding=0)
		)

		self.conv_sce = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
			nn.InstanceNorm2d(channels, affine=True),
			nn.ELU(),
			nn.Conv2d(channels, out_channels_sce, kernel_size=1, stride=1, padding=0)
		)

		self.conv_no = nn.Sequential(
			nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
			nn.InstanceNorm2d(channels, affine=True),
			nn.ELU(),
			nn.Conv2d(channels, out_channels_no, kernel_size=1, stride=1, padding=0)
		)

		self.pool = nn.AdaptiveMaxPool2d((1,channels))
		self.conv_angle = nn.Sequential(
			nn.Conv1d(channels, 128, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.Conv1d(128, self.out_channels_angle, kernel_size=1, stride=1, padding=0)
		)

	def forward(self, embedding, single_repr, coords, backbone_frames, criterion, mask,
				template1, template2, template3, template4, template5, representation=None, bert_mask=None, in_cycle=False):
		if self.e2e:
			if self.model == 'alphafold':
				#if self.embed_type =='msa_transformer':
				if not in_cycle:
					representation = self.msa_project(representation)
					embedding = self.pairwise_project(embedding)
				#print(representation.size())
				if self.msa:
					x, m = self.resnet(embedding, representation,  msa_mask=bert_mask)
				else:
					x, m = self.resnet(embedding, representation)
				#x = x + self.outer_mean(m)
				outs = x.permute(0,3,1,2).contiguous()
			if self.model == 'resnet':
				embedding = embedding.permute(0, 3, 1, 2)
				outs = self.resnet(embedding)
			if self.use_templates:
				temp_out = torch.cat([template1, template2, template3, template4, template5], dim=1)
				temp_out = self.template_project(temp_out)
				outs = outs + temp_out
			
			dist_outs = self.conv_dist(outs)
			mu_outs = self.conv_mu(outs)
			theta_outs = self.conv_theta(outs)
			rho_outs = self.conv_rho(outs)
			sce_outs = self.conv_sce(outs)
			no_outs = self.conv_no(outs)

			#Angle pred
			outs = outs.permute(0,2,3,1)
			pool_outs = self.pool(outs)
			pool_outs = pool_outs.permute(0,3,1,2)
			pool_outs = pool_outs.squeeze(3)
			angle_outs = self.conv_angle(pool_outs)

			phi_outs,psi_outs = angle_outs[:,:int(self.out_channels_angle/2),:],angle_outs[:,int(self.out_channels_angle/2):,:]

			output1 = [dist_outs, mu_outs, theta_outs, rho_outs, sce_outs, no_outs, phi_outs, psi_outs]
			
			m_ = self.single_project(m)
			single = m_[:, 0, ...]
		
			output2 = self.ipa(single + single_repr, x, coords, backbone_frames, criterion, masks=mask, use_fape=True)
			#output2 = self.ipa(single_repr, x, coords, backbone_frames, criterion, masks=mask, use_fape=True)
			if self.msa:
				#self.recycle_embedder(output2[0])
				return output1, output2, self.msa_project2(m), x, m
			return output1, output2, x, m
		#output2 = self.ipa(single_repr, embedding, coords, backbone_frames, criterion, masks=mask, use_fape=True)
		x_true = coords[mask].unsqueeze(0)[:,:,1,:]
		x = self.regression(single_repr)
		#print(x)
		x = x[mask].unsqueeze(0)
		#print(x_true.size())
		#print(x.size())
		#print(x)
		x_ij = x.sub(x_true).norm(p=2, dim=-1)
		
		return None, (x, x_ij.mean()), None, None
	


		



