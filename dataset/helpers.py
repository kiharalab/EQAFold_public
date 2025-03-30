import numpy as np
from os.path import join
import torch

from einops import rearrange


import time
data_dir = "/net/kihara-fast-scratch/jain163/attentivedist2_data"
renum_dir = join(data_dir, "renum_chain")
chain_dir = join(data_dir, "chain")
gt_keys_3d = ['all_atom_positions', 'atom14_gt_positions', 'atom14_alt_gt_positions']
gt_keys_2d = ['all_atom_mask', 'atom14_atom_exists', 'atom14_gt_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 
			'atom37_atom_exists','atom14_alt_gt_exists', 'atom14_atom_is_ambiguous']
gt_frames_keys_3d = ['rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous']
gt_frames_keys_4d = ['torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos']
gt_frames_keys_5d = ['rigidgroups_gt_frames', 'rigidgroups_alt_gt_frames', ]
MT = True
def pad_tensor(batch, lens, dims, key, dtype='float'):
	padded_tensor = torch.zeros(dims)
	if dtype == 'long':
		padded_tensor = torch.zeros(dims).long()
	for i_batch, (data, length) in enumerate(zip(batch, lens)):
		i_data = data[key]
		if len(dims) == 4 and key == 'mt_msa':
			padded_tensor[i_batch, :, :length, :] = i_data
		elif len(dims) == 4:
			padded_tensor[i_batch, :, :length, :length] = i_data
		elif len(dims) == 3 and key == 'embed':
			padded_tensor[i_batch, :, :length] = i_data
		elif len(dims) == 3:
			padded_tensor[i_batch, :length, :length] = i_data
		elif len(dims) == 2:
			padded_tensor[i_batch, :length] = i_data
	return padded_tensor

def pad_tensor2(batch, lens, seq_lens,dims, key, dtype='float'):
	padded_tensor = torch.zeros(dims)
	if dtype == 'long':
		padded_tensor = torch.zeros(dims).long()
	#print('padded tensor: ', padded_tensor.size())
	for i_batch, (data, length, seq_length) in enumerate(zip(batch, lens, seq_lens)):
		i_data = data[key]
		#print('seq_length: ', seq_length)
		padded_tensor[i_batch, :seq_length, :length, :] = i_data
	return padded_tensor

def pad_tensor3(batch, lens, dims, key, dtype='float'):
	padded_tensor = torch.zeros(dims)
	if key == 'aatype':
		padded_tensor.fill_(20)
	if dtype == 'long':
		padded_tensor = torch.zeros(dims).long()
	for i_batch, (data, length) in enumerate(zip(batch, lens)):
		#print(key)
		i_data = data[key]
		if len(dims) == 4 and key == 'coords':
			padded_tensor[i_batch, :length, :, :] = i_data
		elif len(dims) == 4:
			padded_tensor[i_batch, :length, :length, :] = i_data
		elif len(dims) == 3 and (key == 'embed' or key == 'msa_mask'):
			padded_tensor[i_batch, :, :length] = i_data
		elif len(dims) == 3:
			padded_tensor[i_batch, :length, :] = i_data
		elif len(dims) == 2:
			if key == 'mask':
				padded_tensor[i_batch, :length] = i_data
			else:
				padded_tensor[i_batch, :i_data.size(0)] = i_data
	return padded_tensor

def pad_tensor4(batch, lens, dims, key, dtype='float'):
	padded_tensor = torch.zeros(dims)
	if dtype == 'long':
		padded_tensor = torch.zeros(dims).long()
	for i_batch, (data, length) in enumerate(zip(batch, lens)):
		#print(key)
		i_data = data[key]
		if len(dims) == 4:
			padded_tensor[i_batch, :i_data.size(0), :, :] = i_data
		else:
			padded_tensor[i_batch, :i_data.size(0), :] = i_data
	return padded_tensor

def collate_fn(batch, args):
	#t_0 = time.time()
	af2 = args.af2
	collate_dict = {}
	keys_4d_template = ['template1', 'template2', 'template3', 'template4', 'template5']
	keys_3d = ['dist', 'mu', 'rho', 'theta', 'sce', 'no']
	keys_2d = ['phi', 'psi']

	if af2:
		lens = [data['embed'].shape[1] for data in batch]
		#print(lens)
	else:
		lens = [data['embed'].shape[2] for data in batch]
	embed_size = batch[0]['embed'].shape[-1] if af2 else batch[0]['embed'].shape[0]
	batch_size, max_len = (len(batch),max(lens))
	single_size = batch[0]['single_representation'].shape[-1]
	if args.e2e:
		read_size = 508 if af2 else 160#batch[0]['representation'].shape[0]
		representation_size =  batch[0]['representation'].shape[-1]
		seq_lens = [data['representation'].shape[0] for data in batch]

	if args.MT:
		collate_dict['embed2'] = pad_tensor3(batch, lens, [batch_size, max_len, max_len, embed_size], 'embed2')

	if args.use_templates:
		template_feature_size = batch[0]['template1'].shape[0]
		for key in keys_4d_template:
			collate_dict[key] = pad_tensor(batch, lens, [batch_size, template_feature_size, max_len, max_len], key)

	for key in keys_3d:
		if key in batch[0]: #It wont be present for test dataloader
			collate_dict[key] = pad_tensor(batch, lens, [batch_size, max_len, max_len], key, dtype='long')

	for key in keys_2d:
		if key in batch[0]: #It wont be present for test dataloader
			collate_dict[key] = pad_tensor(batch, lens, [batch_size, max_len], key, dtype='long')

	if args.embed =='msa_transformer':
		if af2:
			#print(max_len, lens)
			collate_dict['embed'] = pad_tensor3(batch, lens, [batch_size, max_len, max_len, embed_size], 'embed')
			if args.e2e:
				collate_dict['representation'] = pad_tensor2(batch, lens, seq_lens, [batch_size, read_size, max_len, representation_size], 'representation')
			if args.distill:
				collate_dict['embed_distill'] = pad_tensor3(batch, lens, [batch_size, max_len, max_len, embed_size], 'embed_distill')
				collate_dict['single_distill'] = pad_tensor3(batch, lens, [batch_size, max_len, single_size], 'single_distill')

			collate_dict['single_representation'] = pad_tensor3(batch, lens, [batch_size, max_len, single_size], 'single_representation')
			#collate_dict['coords'] = pad_tensor3(batch, lens, [batch_size, max_len, 3, 3], 'coords')
			#collate_dict['mask'] = pad_tensor3(batch, lens, [batch_size, max_len], 'mask')
			collate_dict['aatype'] = pad_tensor3(batch, lens, [batch_size, max_len], 'aatype', dtype='long')
			collate_dict['residue_index'] = pad_tensor3(batch, lens, [batch_size, max_len], 'residue_index', dtype='long')
			
			for key in gt_keys_3d:
				if 'atom14' in key:
					collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 14, 3], key)
				else:
					collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 37, 3], key)
			for key in gt_keys_2d:
				if key == 'residx_atom37_to_atom14':
					collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 37], key)#, dtype='long')
					continue
				if key == 'residx_atom14_to_atom37':
					collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 14], key, dtype='long')
					continue
				if 'atom14' in key:
					collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 14], key)
				else:
					collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 37], key)
			for key in gt_frames_keys_3d:
				collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 8], key)
			for key in gt_frames_keys_4d:
				collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 7, 2], key)
			for key in gt_frames_keys_5d:
				collate_dict[key] = pad_tensor4(batch, lens, [batch_size, max_len, 8, 4, 4], key)
			collate_dict['torsion_angles_mask'] = pad_tensor4(batch, lens, [batch_size, max_len, 7], 'torsion_angles_mask')
			collate_dict['chi_angles_sin_cos'] = pad_tensor4(batch, lens, [batch_size, max_len, 4, 2], 'chi_angles_sin_cos')
			collate_dict['chi_mask'] = pad_tensor4(batch, lens, [batch_size, max_len, 4], 'chi_mask')
			collate_dict['seq_mask'] = pad_tensor3(batch, lens, [batch_size, max_len], 'seq_mask')
			if args.e2e:
				collate_dict['msa_mask'] = pad_tensor3(batch, lens, [batch_size, read_size, max_len], 'msa_mask')
				collate_dict['pseudo_beta'] = pad_tensor3(batch, lens, [batch_size, max_len, 3], 'pseudo_beta')
				collate_dict['pseudo_beta_mask'] = pad_tensor3(batch, lens, [batch_size, max_len], 'pseudo_beta_mask')

			lens2 = [data['pair_missing'].shape[1] for data in batch]
			max_len2 = max(lens2)
			collate_dict['pair_missing'] = pad_tensor3(batch, lens2, [batch_size, max_len2, max_len2, embed_size], 'pair_missing')
			collate_dict['missing'] = pad_tensor3(batch, lens2, [batch_size, max_len2], 'missing')



			collate_dict['single_missing'] = pad_tensor3(batch, lens2, [batch_size, max_len2, single_size], 'single_missing')
			collate_dict['aatype_missing'] = pad_tensor3(batch, lens2, [batch_size, max_len2], 'aatype_missing', dtype='long')
			collate_dict['seq_mask_missing'] = pad_tensor3(batch, lens2, [batch_size, max_len2], 'seq_mask_missing')

			if args.esm_feats:
				collate_dict['esm'] = pad_tensor3(batch, lens2, [batch_size, max_len2, 33], 'esm')

			if args.rmsf_feats:
				collate_dict['rmsf'] = pad_tensor3(batch, lens2, [batch_size, max_len2, 1], 'rmsf')

			if args.esm_edgefeats:
				# print("in batch pair_missing shape", batch[0]["pair_missing"].shape)
				# print("in batch esm_edge shape", batch[0]["esm_edge"].shape)
				collate_dict['esm_edge'] = pad_tensor3(batch, lens2, [batch_size, max_len2, max_len2, 33], 'esm_edge')

			if args.esm_edgefeats_lastlayer:
				# print("in batch pair_missing shape", batch[0]["pair_missing"].shape)
				# print("in batch esm_edge shape", batch[0]["esm_edge"].shape)
				collate_dict['esm_lastedge'] = pad_tensor3(batch, lens2, [batch_size, max_len2, max_len2, 20], 'esm_lastedge')

			if args.esm_edgefeats_alllayer:
				batched_alllayer = pad_tensor3(batch, lens2, [batch_size, max_len2, max_len2, 33, 20], 'esm_alledge') # pad the batch (BatchXLenthXLengthX33X20)
				batched_alllayer = rearrange(batched_alllayer, " b l w d f -> b (f d) l w") #Group the 33 and 20 dims together
				collate_dict['esm_alledge'] = batched_alllayer

		else:
			collate_dict['embed'] = pad_tensor(batch, lens, [batch_size, embed_size, max_len, max_len], 'embed')
			collate_dict['representation'] = pad_tensor2(batch, lens, seq_lens, [batch_size, read_size, max_len, representation_size], 'representation')
	else:
		collate_dict['embed'] = pad_tensor(batch, lens, [batch_size, embed_size, max_len], 'embed')
		collate_dict['representation'] = pad_tensor2(batch, lens, seq_lens, [batch_size, read_size, max_len, representation_size], 'representation')

	targets = []
	sequences = [] #Jake - Collect the list of sequences in each batch
	record_lines = []
	resolution = []
	seq_length = list()
	if args.hhbond:
		lens0 = [data['hhbond'].shape[0] for data in batch]
		lens1 = [data['hhbond'].shape[1] for data in batch]
		max_len0, max_len1 = max(lens0), max(lens1)

		padded_tensor = torch.zeros([batch_size, max_len0, max_len1])
		padded_tensor2 = torch.zeros([batch_size, max_len0, max_len1])
		padded_tensor3 = torch.zeros([batch_size, max_len0, max_len1])
		for i_batch, (data, length) in enumerate(zip(batch, lens)):
			#print(key)
			i_data = data['hhbond']
			padded_tensor[i_batch, :i_data.size(0), :i_data.size(1)] = i_data
			i_data = data['hydro']
			padded_tensor2[i_batch, :i_data.size(0), :i_data.size(1)] = i_data
			i_data = data['ionic']
			padded_tensor3[i_batch, :i_data.size(0), :i_data.size(1)] = i_data
	
		collate_dict['hhbond'] = padded_tensor
		collate_dict['hydro'] = padded_tensor2
		collate_dict['ionic'] = padded_tensor3
	for i_batch, (data, length) in enumerate(zip(batch, lens)):
		target = data["target"]
		sequence = data["target_seq"] #Jake
		res = data['resolution']
		targets.append(target)
		sequences.append(sequence) #Jake
		resolution.append(res)
		record_lines.append(data['record_lines'])

		seq_length.append(data['seq_length'])

	collate_dict['sequences'] = sequences
	#print(seq_length)
	collate_dict['res_dict'] = data['res_dict']
	#collate_dict['transformer'] = data['transformer']
	collate_dict['record_lines'] = record_lines
	collate_dict['resolution'] = torch.stack(resolution)
	collate_dict['seq_length'] = torch.tensor(seq_length)
	if args.msa:
		collate_dict['msa_mask'] = data['msa_mask']
	if args.test:
		collate_dict['test'] = data['test']
	if args.test_only:
		collate_dict['test'] = data['test']
		######################## JAKES ADDITIONS  ###############################
		collate_dict["target_seq"] = data["target_seq"]  #JAKE ADDING FASTA SEQ
	#collate_dict["targets"] = targets  #JAKE ADDING TARGETS
	# t_diff = time.time() - t_0
	# print(f'finish padding: {t_diff}')
	# print("collate_keys are ", collate_dict.keys())
	# exit()
	return collate_dict, targets

def get_chain_nums(pdb):
	#Returns the information for first 10 residues in the pdb file
	nums = []
	count = 0
	residue_seq_num = ''
	with open(pdb) as p:
		for line in p:
			line = line.strip()
			if line[0:4] == 'ATOM':
				if line[22:26] != residue_seq_num:
					atom_serial_num = line[6:11]
					atom_name = line[12:16]
					residue_name = line[17:20]
					residue_seq_num = line[22:26]
					nums.append([int(residue_seq_num), int(atom_serial_num), atom_name, residue_name])
					count += 1
			if count == 10:
				break
	return nums

def get_chain_2_renum(target):
	#Returns the difference between amino acid numbers of pdb chain and renumbered chain
	chain_pdb = join(chain_dir, target+'.pdb')
	renum_pdb = join(renum_dir, target+'.pdb')
	c_nums = get_chain_nums(chain_pdb)
	r_nums = get_chain_nums(renum_pdb)

	if len(r_nums) == 0: #No residues in renum pdb
		return 0

	c_pos = 0
	r_pos = 0
	while 1:
		c_residue_seq_num, c_atom_serial_num, c_atom_name, c_residue_name = c_nums[c_pos]
		r_residue_seq_num, r_atom_serial_num, r_atom_name, r_residue_name = r_nums[r_pos]
		if c_atom_serial_num == r_atom_serial_num and c_atom_name == r_atom_name and c_residue_name == r_residue_name:
			difference = c_residue_seq_num - r_residue_seq_num
			return difference
		else:
			if r_atom_serial_num > c_atom_serial_num:
				c_pos += 1
			else:
				r_pos += 1

def get_onehot_encoding(sequence):
	aa = "ARNDCQEGHILKMFPSTWYV"
	seq_feature = np.zeros((20, len(sequence)))
	for i in range(len(sequence)):
		for j in range(len(aa)):
			if sequence[i] == aa[j]:
				seq_feature[j][i] = 1
	return seq_feature
