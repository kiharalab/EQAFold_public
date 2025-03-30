import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock2D, ResNet

from .alphafold2 import *
from .invariant_point_attention import *

class AttentiveDist5(nn.Module):
	def __init__(self, args):
		super(AttentiveDist5, self).__init__()

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
			if self.e2e:
				self.msa_project = nn.Sequential(nn.Linear(256, evo_dim))
				self.pairwise_project = nn.Sequential(nn.Linear(128, evo_dim))
			'''
			self.msa_project = nn.Sequential(nn.Linear(768, 384))
			self.pairwise_project = nn.Sequential(nn.Linear(144, 128))
			'''
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
		pairwise_repr_dim_default =64 if self.e2e else 128
		self.ipa = IPATransformer(dim=384, num_tokens = 21, depth=args.ipa_depth, require_pairwise_repr = True,
							 predict_points = True, pairwise_repr_dim_default=pairwise_repr_dim_default, args=args)
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

	def forward_one_cycle(self, embedding, single_repr, coords, backbone_frames, criterion, mask,
				template1, template2, template3, template4, template5, representation=None, bert_mask=None, in_cycle=False, aatype=None, gt_frames=None, atom14=None):
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
		#single_repr = torch.randn_like(single_repr)
		#representation = self.msa_project(representation)
		#embedding = self.pairwise_project(embedding.permute(0, 2, 3, 1))
		output2 = self.ipa(single_repr, embedding, coords, backbone_frames, criterion, masks=mask, use_fape=True, aatype=aatype, gt_frames=gt_frames, atom14=atom14)
		#output2 = self.ipa(representation[:, 0, ...], embedding, coords, backbone_frames, criterion, masks=mask, use_fape=True)
		return None, output2, None, None

	def recycle_embedder(self, coords):
		batch, length, dim = coords.size()
		#dist = torch.ones(length, length)
		dist2 = coords[0].unsqueeze(2).repeat(1,1,length)
		#print(dist2.permute(2, 1, 0).size())
		if self.use_cuda:
			#dist = dist.cuda(self.device_id)
			dist2 = dist2.cuda(self.device_id)
		'''
		for i in range(length):
			for j in range(length):
				dist[i][j] = (coords[0][i] - coords[0][j]).norm()
		'''
		dist2 = torch.norm((dist2 - dist2.permute(2, 1, 0)), dim=1)
		no_bins = torch.arange(3.375, 21.375, 1.25).unsqueeze(0).unsqueeze(0)
		if self.use_cuda:
			#dist = dist.cuda(self.device_id)
			no_bins = no_bins.cuda(self.device_id)
		no_bins = no_bins.repeat(length, length, 1)
		#print(dist2)
		dist2 = dist2.unsqueeze(2).repeat(1, 1, 15)
		dist_2_bins = torch.argmin(torch.abs(dist2 - no_bins), dim=-1)

		dist_recycle = F.one_hot(dist_2_bins, num_classes=15)		
		return dist_recycle
	
	def forward_recycle(self, embedding, single_repr, coords, backbone_frames, criterion, mask,
			template1, template2, template3, template4, template5, representation=None, bert_mask=None, true_mask=None):
		recycle = 4
		with torch.no_grad():
			if self.msa:
				outputs1, outputs2, msa, pair, msa_repr = self.forward_one_cycle(embedding, single_repr, coords, backbone_frames, criterion, mask,
						template1, template2, template3, template4, template5, representation, bert_mask)
			else:
				outputs1, outputs2, pair, msa_repr = self.forward_one_cycle(embedding, single_repr, coords, backbone_frames, criterion, mask,
						template1, template2, template3, template4, template5, representation)
			msa_representation = msa_repr
			for  i in range(recycle-2):
				points, fape = outputs2
				if self.use_cuda:
					points = points.cuda(self.device_id).float()
				points = self.recycle_embedder(points)
				pair = self.norm1(pair) + self.point_project(points.float())
				msa_representation[:, 0, ...] = self.norm2(msa_repr[:, 0, ...]) 
				if self.msa:
					outputs1, outputs2, msa, pair, msa_repr = self.forward_one_cycle(pair, single_repr, coords, backbone_frames, criterion, mask,
							template1, template2, template3, template4, template5, msa_representation, bert_mask, in_cycle=True)
				else:
					outputs1, outputs2, pair, msa_repr = self.forward_one_cycle(pair, single_repr, coords, backbone_frames, criterion, mask,
							template1, template2, template3, template4, template5, msa_representation, in_cycle=True)

		if self.msa:
			outputs1, outputs2, msa, pair, msa_repr = self.forward_one_cycle(pair, single_repr, coords, backbone_frames, criterion, mask,
					template1, template2, template3, template4, template5, msa_representation, bert_mask, in_cycle=True)
			return outputs1, outputs2, msa, None, None
		else:
			outputs1, outputs2, pair, msa_repr = self.forward_one_cycle(pair, single_repr, coords, backbone_frames, criterion, mask,
					template1, template2, template3, template4, template5, msa_representation, in_cycle=True)
			return outputs1, outputs2, None, None
	
	def forward(self, embedding, single_repr, coords, backbone_frames, criterion, mask,
		template1, template2, template3, template4, template5, representation=None, bert_mask=None, aatype=None, gt_frames=None, atom14=None):
		if self.recycle:
			return self.forward_recycle(embedding, single_repr, coords, backbone_frames, criterion, mask,
					template1, template2, template3, template4, template5, representation, bert_mask)
		else:
			return self.forward_one_cycle(embedding, single_repr, coords, backbone_frames, criterion, mask,
					template1, template2, template3, template4, template5, representation, bert_mask, aatype=aatype, gt_frames=gt_frames, atom14=atom14)

		



