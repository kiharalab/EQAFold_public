import numpy as np
import random
import re
import torch
import torch.nn as nn
import json
import os
from os.path import join,exists
from torch.autograd import Variable
from torch import FloatTensor, LongTensor
from torch.utils.data.dataset import Dataset

from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

import pickle as pkl
#from tape import ProteinBertModel, TAPETokenizer

from .helpers import get_chain_2_renum, get_onehot_encoding
import sys

from .af2_util import protein, all_atom

from typing import Mapping, Optional, Sequence, Any
from dataset.openfold_util import protein as pr
from dataset.openfold_util import data_transforms, residue_constants
import jax.numpy as jnp
from typing import Dict
from einops import rearrange

import time
msa_transformer_dir_3k='/kihara-fast-scratch-2/data/zhan1797/attentivedist2_data/msa_transformer_embeddings_3k'
def ccmpred_parser(filename):
    data = np.loadtxt(filename)
    n, _ = data.shape
    matrix  = np.zeros((1,n,n),dtype = "float32")                       
    matrix[0, :, :] = data[:, :]
    return matrix

def get_seq(seq_file):
    with open(seq_file, 'r') as s:
        s.readline() #read the '>' line
        seq = ""
        for line in s:
            seq += line.strip()
    return seq

def get_largest_domain(domain_file):
    with open(domain_file, 'r') as d:
        domains_str = d.readline().rstrip()
    domains = re.split(r'[;,]', domains_str)

    max_len = 0
    start,end = 0, 0
    for domain in domains:
        if len(domain.split('-')) == 2:
            cur_start, cur_end = domain.split('-')
            domain_len = int(cur_end) - int(cur_start)
            if domain_len > max_len:
                max_len = domain_len
                start = int(cur_start)
                end = int(cur_end)
    return start, end, (end - start + 1)

def T_to_Q_aligned_matrix(empty_query_mtx, template_mtx, query_2_template_idx):
    query_mtx = empty_query_mtx
    for i in range(len(query_mtx)):
        for j in range(len(query_mtx[0])):
            query_row_idx = i
            query_col_idx = j
            if query_row_idx in query_2_template_idx and query_col_idx in query_2_template_idx:
                template_row_idx = query_2_template_idx[query_row_idx]
                template_col_idx = query_2_template_idx[query_col_idx]
                query_mtx[query_row_idx][query_col_idx] = template_mtx[template_row_idx][template_col_idx]
    return query_mtx

def replace_special_aa(seq):
    seq = seq.replace('O','K')
    seq = seq.replace('U','C')
    return seq

def aligned_template_to_template_fasta(template_seq, template_aligned_seq, template_start_index):
    template_seq = replace_special_aa(template_seq)
    template_aligned_seq = replace_special_aa(template_aligned_seq)
    
    blosum62 = substitution_matrices.load('BLOSUM62')
    alignments = pairwise2.align.globalds(template_seq, template_aligned_seq, blosum62, -10, -0.5)
    template_fasta_pairwise = alignments[0][0]
    template_aligned_pairwise = alignments[0][1]
    aligned_to_fasta = {}

    aligned_template_index = int(template_start_index) - 1
    fasta_index = 0
    different_aa_count = 0

    for i in range(len(template_aligned_pairwise)):
        if template_aligned_pairwise[i] != '-' and template_fasta_pairwise[i] != '-':
            if template_aligned_pairwise[i] != template_fasta_pairwise[i]:
                different_aa_count += 1
            aligned_to_fasta[aligned_template_index] = fasta_index
            aligned_template_index += 1
            fasta_index += 1
        elif template_aligned_pairwise[i] == '-':
            fasta_index += 1
        elif template_fasta_pairwise[i] == '-':
            aligned_template_index += 1

    #If the matched template fasta sequence is more than 30% different then template aligned sequnce, then dont use it
    if different_aa_count/float(len(aligned_to_fasta)) > 0.3:
        # print("Mismatch percentage is ", different_aa_count/float(len(aligned_to_fasta)))
        return {}

    return aligned_to_fasta

def get_template_feature(template_info, template_chain_dir, target):
    #Length of query protein from which we are finding the template
    query_length = template_info['query_length']

    #Template PDB information
    template_name = template_info["name"].replace("_","")
    template_chain_out_dir = join(template_chain_dir,template_name)
    template_distance_bins = np.load(join(template_chain_out_dir, template_name+'_dist_bin.npy'))
    template_sce_bins = np.load(join(template_chain_out_dir, template_name+'_sce_bin.npy'))

    #Template alignment with query
    Q = template_info['Q']
    T = template_info['T']
    T_ss_dssp = template_info['T_ss_dssp']
    query_range = template_info['query_range']
    template_range = template_info['template_range']

    query_2_template_index = {}
    query_start_index = int(query_range.split('-')[0])
    template_start_index, template_end_index = int(template_range.split('-')[0]), int(template_range.split('-')[1])

    #Because the pdb70 template seqeunce and the template fasta sequences for some cases are different, we need to correct the template range
    #based on the template fasta.
    template_aligned_seq = T.replace("-","")
    template_seq_file = join(template_chain_out_dir,template_name+'.fasta')
    with open(template_seq_file, 'r') as s:
            s.readline() #read the '>' line
            template_seq = ""
            for line in s:
                template_seq += line.strip()

    #Search for aligned template in the template fasta
    res = re.search(template_aligned_seq, template_seq)
    if res:
        #If aligned template seq is found, update the position if necessary
        template_fasta_not_matched = False
        matched_start_pos = res.start() + 1
        matched_end_pos = res.end()
        if template_start_index != matched_start_pos:
            template_start_index = matched_start_pos
    else:
        #If aligned template seq is not found, compute which residue of aligned template corresponds to template fasta
        template_fasta_not_matched = True
        template_aligned_to_fasta = aligned_template_to_template_fasta(template_seq, template_aligned_seq, template_start_index)

    query_index = int(query_start_index) - 1
    template_index = int(template_start_index) - 1
    for i in range(len(Q)):
        if Q[i] != '-' and T[i] != '-':
            if template_fasta_not_matched:
                if template_index in template_aligned_to_fasta:
                    query_2_template_index[query_index] = template_aligned_to_fasta[template_index]
            else:
                query_2_template_index[query_index] = template_index
            query_index += 1
            template_index += 1
        elif Q[i] == '-':
            template_index += 1
        elif T[i] == '-':
            query_index += 1

    #Get template features
    num_distance_bins = 21
    query_distance_bins_empty = np.full((query_length, query_length), num_distance_bins-1) #No info bin = num_distance_bins-1
    query_distance_bins_aligned = T_to_Q_aligned_matrix(query_distance_bins_empty, template_distance_bins, query_2_template_index)
    query_distance_bins_aligned = torch.from_numpy(query_distance_bins_aligned).type(torch.LongTensor)
    query_distance_bins_onehot = torch.nn.functional.one_hot(query_distance_bins_aligned,  num_classes=num_distance_bins)
    query_distance_bins_onehot = np.swapaxes(query_distance_bins_onehot, 2, 0)
    
    num_sce_bins = 39
    query_sce_bins_empty = np.full((query_length, query_length), num_sce_bins-1) #No info bin = num_sce_bins-1
    query_sce_bins_aligned = T_to_Q_aligned_matrix(query_sce_bins_empty, template_sce_bins, query_2_template_index)
    query_sce_bins_aligned = torch.from_numpy(query_sce_bins_aligned).type(torch.LongTensor)
    query_sce_bins_onehot = torch.nn.functional.one_hot(query_sce_bins_aligned,  num_classes=num_sce_bins)
    query_sce_bins_onehot = np.swapaxes(query_sce_bins_onehot, 2, 0)

    identity_feature = torch.unsqueeze(torch.full((query_length, query_length), template_info["identity"]), 0).type(torch.LongTensor)
    similarity_feature = torch.unsqueeze(torch.full((query_length, query_length), template_info["similarity"]), 0).type(torch.LongTensor)
    prob_feature = torch.unsqueeze(torch.full((query_length, query_length), template_info["prob"]), 0).type(torch.LongTensor)
    coverage_feature = torch.unsqueeze(torch.full((query_length, query_length), template_info["coverage"]), 0).type(torch.LongTensor)

    template_feature = torch.cat([query_distance_bins_onehot, query_sce_bins_onehot, identity_feature, 
                        similarity_feature, prob_feature, coverage_feature], dim=0)

    return template_feature
'''open fold feature'''
#========================================================================================
def aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]] 
        for i in range(len(aatype))
    ])

FeatureDict = Mapping[str, np.ndarray]
def make_protein_features(
    protein_object: protein.Protein, 
    description: str, 
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )
    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask
    
    pdb_feats["all_atom_positions"] = all_atom_positions
    pdb_feats["all_atom_mask"] = all_atom_mask
    
    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(1.).astype(np.float32)

    return pdb_feats

def make_sequence_features(
    sequence: str, description: str='', num_res: int=0
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    '''
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    '''
    return features

def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
    confidence_threshold: float = 0.5,
) -> FeatureDict:
    pdb_feats = make_protein_features(protein_object, description)
    high_confidence = protein_object.b_factors > confidence_threshold
    high_confidence = np.any(high_confidence, axis=-1)
    for i, confident in enumerate(high_confidence):
        if(not confident):
            pdb_feats["all_atom_mask"][i] = 0
    
    return pdb_feats
#========================================================================================
def squared_difference(x, y):
  return torch.square(torch.tensor(x) - torch.tensor(y))

def find_optimal_renaming(
    atom14_gt_positions: jnp.ndarray,  # (N, 14, 3)
    atom14_alt_gt_positions: jnp.ndarray,  # (N, 14, 3)
    atom14_atom_is_ambiguous: jnp.ndarray,  # (N, 14)
    atom14_gt_exists: jnp.ndarray,  # (N, 14)
    atom14_pred_positions: jnp.ndarray,  # (N, 14, 3)
    atom14_atom_exists: jnp.ndarray,  # (N, 14)
) -> jnp.ndarray:  # (N):
  """Find optimal renaming for ground truth that maximizes LDDT.

  Jumper et al. (2021) Suppl. Alg. 26
  "renameSymmetricGroundTruthAtoms" lines 1-5

  Args:
    atom14_gt_positions: Ground truth positions in global frame of ground truth.
    atom14_alt_gt_positions: Alternate ground truth positions in global frame of
      ground truth with coordinates of ambiguous atoms swapped relative to
      'atom14_gt_positions'.
    atom14_atom_is_ambiguous: Mask denoting whether atom is among ambiguous
      atoms, see Jumper et al. (2021) Suppl. Table 3
    atom14_gt_exists: Mask denoting whether atom at positions exists in ground
      truth.
    atom14_pred_positions: Predicted positions of atoms in
      global prediction frame
    atom14_atom_exists: Mask denoting whether atom at positions exists for given
      amino acid type

  Returns:
    Float array of shape [N] with 1. where atom14_alt_gt_positions is closer to
    prediction and 0. otherwise
  """
  assert len(atom14_gt_positions.shape) == 3
  assert len(atom14_alt_gt_positions.shape) == 3
  assert len(atom14_atom_is_ambiguous.shape) == 2
  assert len(atom14_gt_exists.shape) == 2
  assert len(atom14_pred_positions.shape) == 3
  assert len(atom14_atom_exists.shape) == 2

  # Create the pred distance matrix.
  # shape (N, N, 14, 14)
  pred_dists = torch.sqrt(1e-10 + torch.sum(
      squared_difference(
          atom14_pred_positions[:, None, :, None, :],
          atom14_pred_positions[None, :, None, :, :]),
      axis=-1))

  # Compute distances for ground truth with original and alternative names.
  # shape (N, N, 14, 14)
  gt_dists = torch.sqrt(1e-10 + torch.sum(
      squared_difference(
          atom14_gt_positions[:, None, :, None, :],
          atom14_gt_positions[None, :, None, :, :]),
      axis=-1))
  alt_gt_dists = torch.sqrt(1e-10 + torch.sum(
      squared_difference(
          atom14_alt_gt_positions[:, None, :, None, :],
          atom14_alt_gt_positions[None, :, None, :, :]),
      axis=-1))

  # Compute LDDT's.
  # shape (N, N, 14, 14)
  lddt = torch.sqrt(1e-10 + squared_difference(pred_dists, gt_dists))
  alt_lddt = torch.sqrt(1e-10 + squared_difference(pred_dists, alt_gt_dists))

  # Create a mask for ambiguous atoms in rows vs. non-ambiguous atoms
  # in cols.
  # shape (N ,N, 14, 14)
  mask = (atom14_gt_exists[:, None, :, None] *  # rows
          atom14_atom_is_ambiguous[:, None, :, None] *  # rows
          atom14_gt_exists[None, :, None, :] *  # cols
          (1. - atom14_atom_is_ambiguous[None, :, None, :]))  # cols

  mask = torch.tensor(mask)
  # Aggregate distances for each residue to the non-amibuguous atoms.
  # shape (N)
  per_res_lddt = torch.sum(mask * lddt, axis=[1, 2, 3])
  alt_per_res_lddt = torch.sum(mask * alt_lddt, axis=[1, 2, 3])

  # Decide for each residue, whether alternative naming is better.
  # shape (N)
  alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).to(torch.float32)

  return alt_naming_is_better  # shape (N)

def compute_renamed_ground_truth(
    batch: Dict[str, jnp.ndarray],
    atom14_pred_positions: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
  """Find optimal renaming of ground truth based on the predicted positions.

  Jumper et al. (2021) Suppl. Alg. 26 "renameSymmetricGroundTruthAtoms"

  This renamed ground truth is then used for all losses,
  such that each loss moves the atoms in the same direction.
  Shape (N).

  Args:
    batch: Dictionary containing:
      * atom14_gt_positions: Ground truth positions.
      * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
      * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
          renaming swaps.
      * atom14_gt_exists: Mask for which atoms exist in ground truth.
      * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
          after renaming.
      * atom14_atom_exists: Mask for whether each atom is part of the given
          amino acid type.
    atom14_pred_positions: Array of atom positions in global frame with shape
      (N, 14, 3).
  Returns:
    Dictionary containing:
      alt_naming_is_better: Array with 1.0 where alternative swap is better.
      renamed_atom14_gt_positions: Array of optimal ground truth positions
        after renaming swaps are performed.
      renamed_atom14_gt_exists: Mask after renaming swap is performed.
  """
  alt_naming_is_better = find_optimal_renaming(
      atom14_gt_positions=batch['atom14_gt_positions'],
      atom14_alt_gt_positions=batch['atom14_alt_gt_positions'],
      atom14_atom_is_ambiguous=batch['atom14_atom_is_ambiguous'],
      atom14_gt_exists=batch['atom14_gt_exists'],
      atom14_pred_positions=atom14_pred_positions,
      atom14_atom_exists=batch['atom14_atom_exists'])

  renamed_atom14_gt_positions = (
      (1. - alt_naming_is_better[:, None, None])
      * batch['atom14_gt_positions']
      + alt_naming_is_better[:, None, None]
      * batch['atom14_alt_gt_positions'])

  renamed_atom14_gt_mask = (
      (1. - alt_naming_is_better[:, None]) * batch['atom14_gt_exists']
      + alt_naming_is_better[:, None] * batch['atom14_alt_gt_exists'])

  return {
      'alt_naming_is_better': alt_naming_is_better,  # (N)
      'renamed_atom14_gt_positions': renamed_atom14_gt_positions,  # (N, 14, 3)
      'renamed_atom14_gt_exists': renamed_atom14_gt_mask,  # (N, 14)
  }

def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    is_gly = torch.eq(aatype, residue_constants.restype_order["G"])
    ca_idx = residue_constants.atom_order["CA"]
    cb_idx = residue_constants.atom_order["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta

msa_transformer_dir = '/kihara-fast-scratch-2/data/zhan1797/attentivedist2_data/msa_transformer_embeddings'
#target validation set
msa_transformer_dir = './val'
#swap_embedding_dir = '/fast-scratch-3/data/zhan1797/swap_msa_embeddings'
distill_dir = '/fast-scratch-3/data/zhan1797/embeddings_model1'
class AttentiveDistDataset(Dataset):

    def __init__(self, targets_file, feature_dir, label_dir, seq_dir, max_len, datatype='train',
                 use_templates=True, template_dir=None, template_chain_dir=None, train_size=0, 
                 embed='onehot', full=False, msa_embedding_dir=None, af2=True, msa=False, e2e=False,
                 hhbond=False, args=None):

        with open(targets_file, 'r') as f:
            self.targets = f.read().splitlines()
            if datatype == 'train' and train_size != 0:
                self.targets = self.targets[:train_size]
        #self.targets = self.targets[:2]
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.seq_dir = seq_dir
        self.datatype = datatype
        self.max_len = max_len
        self.use_templates = use_templates
        self.template_dir = template_dir
        self.template_chain_dir = template_chain_dir
        self.embed = embed

        self.full = full
        self.msa_embedding_dir = msa_embedding_dir
        self.af2 = af2
        self.msa = msa
        self.e2e = e2e

        self.hhbond = hhbond
        self.args = args
        # with open('/net/kihara/home/zhan1797/Desktop/af2_e2e/swap_targets', 'r') as f:
        #     swap_targets = f.read().splitlines()
        swap_targets = []
        self.swap_targets = swap_targets
        self.distill = args.distill
        self.esm_feats = args.esm_feats
        self.rmsf_feats = args.rmsf_feats
        self.esm_edgefeats = args.esm_edgefeats
        self.esm_edgefeats_lastlayer = args.esm_edgefeats_lastlayer
        self.esm_edgefeats_alllayer = args.esm_edgefeats_alllayer

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        
        target = self.targets[index]
        # t_0 = time.time()

        # if target in self.swap_targets and self.datatype == "train":
        #    msa_embedding_dir = join(swap_embedding_dir, target)
        # else:
        #    msa_embedding_dir = join(self.msa_embedding_dir, target)
        
        msa_embedding_dir = join(self.msa_embedding_dir, target)
        
        target_seq = get_seq(join(msa_embedding_dir,target+'.fasta'))
        
        target_seq_len = len(target_seq)
        target_feature_dir = join(self.feature_dir, target)
        target_label_dir = join(self.label_dir, target)

        # print(target, target_seq_len)
        if self.datatype == "train":
            crop_size = self.args.max_len
            g = torch.Generator(device=torch.device('cpu'))
            if self.args.seed is not None:
                g.manual_seed(self.args.seed)
            num_res_crop_size = min(int(target_seq_len), crop_size)

            def _randint(lower, upper):
                return int(torch.randint(
                        lower,
                        upper + 1,
                        (1,),
                        device=torch.device('cpu'),
                        generator=g,
                )[0])
            
            n = target_seq_len - num_res_crop_size
            right_anchor = n
            num_res_crop_start = _randint(0, right_anchor)
            start_position = num_res_crop_start
            end_position = num_res_crop_start + crop_size
            #print(f'{target} crop start: ', start_position)
            #print(f'{target} crop end: ',end_position)
        else:
            start_position = 0
            end_position = target_seq_len

        mask_crop = torch.zeros(target_seq_len)
        mask_crop[start_position: end_position] = 1
       
        data = {}
        data["target"] = target
        msa_rows = 508
        
        #emd_dir = join(msa_embedding_dir, f'model_1.npz')
        #emd = np.load(emd_dir)
        #try:
        #for alphafold
        if self.datatype == 'test':
            emd_dir = join(msa_embedding_dir, f'model_1.npz')
        else:
            emd_dir = join(msa_embedding_dir, f'model_1.npz')

        if self.msa:
            feature_dir = join(msa_embedding_dir, f'{target}/model_1_feature.pkl')
            features = None
            with open(feature_dir, 'rb') as f:
                features = pkl.load(f)

            bert_mask = features['bert_mask']
            true_msa = features['true_msa']
            
            bert_mask = bert_mask[-1][:msa_rows, start_position:end_position]
            true_msa = true_msa[-1][:msa_rows, start_position:end_position]
            
            bert_mask = torch.tensor(bert_mask)
            true_msa = torch.tensor(true_msa, dtype=torch.int64)
            data['msa_mask'] = [bert_mask, true_msa]
        emd = np.load(emd_dir)

        if self.args.MT:
            emd_dir2 = '/kihara-fast-scratch-2/data/zhan1797/embeddings_alphafold2_attentivedist'
            emd_dir2 = join(emd_dir2, target)
            emd_dir2 = join(emd_dir2, f'{target}/model_1.npz')
            emd2 = np.load(emd_dir2)
            sequence_output2 = emd2['pair']
            sequence_output2 = torch.tensor(sequence_output2)
        sequence_output = emd['pair']
        #sequence_representation = emd['msa'][:msa_rows, ...]
        single = emd['single']
        a = torch.tensor(single)
        b = torch.tensor(sequence_output)
        
        mask_a = a[mask_crop.bool()]
        mask_b = b[mask_crop.bool()]
        mask_b = mask_b.permute(1, 0, 2)
        mask_b = mask_b[mask_crop.bool()]
        mask_b = mask_b.permute(1, 0, 2)

        data['pair_missing'] = mask_b
        data['single_missing'] = mask_a
            #================================================================

        single = torch.tensor(single)
        sequence_output = torch.tensor(sequence_output)
        if self.e2e:
            sequence_representation = emd['msa'][:msa_rows, ...]
            sequence_representation = torch.tensor(sequence_representation)

        if self.datatype == 'test':
            single_test, pair_test = single.unsqueeze(0), sequence_output.unsqueeze(0)
            protein_test = make_sequence_features(target_seq, num_res=target_seq_len)
            features_test = np.argmax(protein_test['aatype'], axis=1)
            protein_test['aatype'] = features_test
            protein_test = all_atom.make_atom14_positions(protein_test, training=False)
            for key in protein_test:
                #print(f'{key}, ', protein_test[key].shape)
                protein_test[key] = torch.tensor(protein_test[key]).unsqueeze(0)
                #print(protein_test[key].size())
            data['test'] = {
                'single': single_test,
                'pair': pair_test,
                'aatype': torch.tensor(features_test).unsqueeze(0),
                'protein': protein_test
            }
            if self.args.MT:
                data['test'].update({'emd2': sequence_output2})

        protein_test = make_sequence_features(target_seq, num_res=target_seq_len)
        features_test = np.argmax(protein_test['aatype'], axis=1)
        aatype_missing = torch.tensor(features_test)[mask_crop.bool()]
        data['aatype_missing'] = aatype_missing
        seq_mask_missing = torch.ones(aatype_missing.shape, dtype=torch.float32)
        data['seq_mask_missing'] = seq_mask_missing

        resoltion_line = 'REMARK   2 RESOLUTION.'
        #ori_pdb = join(PDB_DIR,target[:-1]+'.pdb')
        ori_pdb = join(msa_embedding_dir, target +'.pdb')
        resolution = 1

        if self.datatype == 'test':
            resolution=1
        else:
            try: #Try default PDB resolution search method by Zicong
                with open(ori_pdb, 'r') as f:
                    for line in f.readlines():
                        if line.startswith(resoltion_line):
                            resolution = float(line[24:30].lstrip())
            except FileNotFoundError as e:  #Maybe only CIF is available - Jake
                ori_cif = join(msa_embedding_dir, target +'.cif')
                #Get the resolution from the cif header, and return the default (1) otherwise
                header_dict = MMCIF2Dict(ori_cif)
                if "_reflns.d_resolution_high" in header_dict.keys():
                    resolution = float(header_dict["_reflns.d_resolution_high"][0]) #Comes in a list for some reason

        data['resolution'] = torch.tensor([resolution])
        def load_file(pdb_file):
            res_dict = {}
            seq_len = target_seq_len
            coords = np.zeros((seq_len,3,3))
            mask = np.zeros(seq_len,dtype=int)
            record_lines =[]
            with open(pdb_file,"r") as f:
                for line in f.readlines():
                    if " N " in line and line.startswith("ATOM"): 
                        # spl = line.strip().split()
                        line = line.strip()
                        n_res_no = int(line[22:26].strip())-1
                        # print("N res no line", line)
                        # if n_res_no<start_position or n_res_no >= end_position: continue
                        mask[n_res_no] = 1 
                        coords[n_res_no][0][0], coords[n_res_no][0][1], coords[n_res_no][0][2] = float(line[26:38].strip()),float(line[38:46].strip()),float(line[46:54].strip())
                    elif " CA " in line and line.startswith("ATOM"): 
                        # spl = line.strip().split()
                        line = line.strip()
                        ca_res_no = int(line[22:26].strip())-1
                        # print("CA res no line", line)
                        # if ca_res_no<start_position or ca_res_no >= end_position: continue
                        # print("JAKE ca stuff", ca_res_no, n_res_no)
                        assert ca_res_no == n_res_no
                        record_lines.append(line)
                        coords[ca_res_no][1][0], coords[ca_res_no][1][1], coords[ca_res_no][1][2] = float(line[26:38].strip()),float(line[38:46].strip()),float(line[46:54].strip())
                    elif " C " in line and line.startswith("ATOM"): 
                        # spl = line.strip().split()
                        line = line.strip()
                        c_res_no = int(line[22:26].strip())-1
                        # if c_res_no<start_position or c_res_no >= end_position: continue
                        assert c_res_no == ca_res_no
                        # print("C line is", line)
                        coords[c_res_no][2][0], coords[c_res_no][2][1], coords[c_res_no][2][2] = float(line[26:38].strip()),float(line[38:46].strip()),float(line[46:54].strip())
            #coords = coords[:len(record_lines),...]
            #mask = mask[:len(record_lines),...]
            res_dict["coords"] = torch.tensor(coords[start_position:end_position, ...])
            aatype_start = np.sum(mask[:start_position])
            
            mask_tmp = torch.tensor(mask)
            res_dict["mask"] = torch.tensor(mask[start_position:end_position, ...])
            
            res_dict["record_lines"] = record_lines[aatype_start:aatype_start+torch.sum(res_dict["mask"])]
            return res_dict, aatype_start, mask_tmp

        pdb = join(msa_embedding_dir,target+'_renum.pdb')
        pdb_str = ''

        # if self.datatype != 'test':
        with open(pdb,"r") as f:
            pdb_str = f.read()
        prot = protein.from_pdb_string(pdb_str)

        ca = prot.atom_positions[:, 1, :]
        #raw_chain_idx = prot.chain_index
        pdb_length = ca.shape[0]
        res_dict, aatype_start, mask_tmp = load_file(pdb)
        mask_pdb = torch.zeros(pdb_length)
        mask_st = torch.sum(mask_tmp[:start_position])
        mask_len = torch.sum(mask_tmp[start_position: end_position])
        mask_pdb[mask_st: mask_st + mask_len] = 1

        res_dict['single_repr'] = single
        data['res_dict'] = res_dict


        data['missing'] = mask_tmp[mask_crop.bool()]
        data['single_representation'] = single
        #data['embed'] = sequence_output.unsqueeze(0)[mask_tmp]
        a = single#.unsqueeze(0)
        b = sequence_output#.unsqueeze(0)
        try:
            #first mask missing
            mask_a = a[mask_tmp.bool()]
            mask_b = b[mask_tmp.bool()]
            mask_b = mask_b.permute(1, 0, 2)#.unsqueeze(0)
            mask_b = mask_b[mask_tmp.bool()]
            mask_b = mask_b.permute(1, 0, 2)
            #first then crop
            mask_a = mask_a[mask_pdb.bool()]
            mask_b = mask_b[mask_pdb.bool()]
            mask_b = mask_b.permute(1, 0, 2)#.unsqueeze(0)
            mask_b = mask_b[mask_pdb.bool()]
            mask_b = mask_b.permute(1, 0, 2)
        except:
            #first mask missing
            mask_a = a[mask_crop.bool()]
            mask_b = b[mask_crop.bool()]
            mask_b = mask_b.permute(1, 0, 2)#.unsqueeze(0)
            mask_b = mask_b[mask_crop.bool()]
            mask_b = mask_b.permute(1, 0, 2) 

        if self.esm_feats:
            #Add ESM Embeddings
            esm_dir = join(msa_embedding_dir, f"{target}_esm_layer.npz") #Full path
            esm_data = torch.tensor(np.load(esm_dir)["arr_0"]) #Load as L*33 matrix
            data["esm"] = esm_data[mask_crop.bool()] #Apply mask cropping

        if self.rmsf_feats:
            rmsf_dir = join(msa_embedding_dir, f"{target}_rmsf.npz") #Full path
            rmsf_data = torch.tensor(np.load(rmsf_dir)["arr_0"]) #Load as L*320 matrix
            rmsf_data = rmsf_data.reshape(-1, 1)  #LX1 (To match LX33)
            # print("rmsf_shape is", rmsf_data.shape)
            
            data["rmsf"] = rmsf_data[mask_crop.bool()] #Apply mask cropping
            # print("rmsf_shape is now", data["rmsf"].shape)


        if self.esm_edgefeats:
            esm_edge_dir = join(msa_embedding_dir, f"{target}_esm_layerattn.npz") #Full path
            esm_edge_data = torch.tensor(np.load(esm_edge_dir)["arr_0"]) #Load as LXLX33 matrix
            esm_edge_data = rearrange(esm_edge_data, "f w l -> w l f")
            #Process copied from pair_missing process above
            esm_edge_data = esm_edge_data[mask_crop.bool()]
            esm_edge_data = esm_edge_data.permute(1, 0, 2)
            esm_edge_data = esm_edge_data[mask_crop.bool()]
            esm_edge_data = esm_edge_data.permute(1, 0, 2)
            data["esm_edge"] = esm_edge_data

        if self.esm_edgefeats_lastlayer:
            esm_lastedge_dir = join(msa_embedding_dir, f"{target}_esm_lastattn.npz") #Full path
            esm_lastedge_data = torch.tensor(np.load(esm_lastedge_dir)["arr_0"]) #Load as LXLX20 matrix
            esm_lastedge_data = rearrange(esm_lastedge_data, "f w l -> w l f")
            #Process copied from pair_missing process above
            esm_lastedge_data = esm_lastedge_data[mask_crop.bool()]
            esm_lastedge_data = esm_lastedge_data.permute(1, 0, 2)
            esm_lastedge_data = esm_lastedge_data[mask_crop.bool()]
            esm_lastedge_data = esm_lastedge_data.permute(1, 0, 2)
            data["esm_lastedge"] = esm_lastedge_data


        if self.esm_edgefeats_alllayer:
            esm_alledge_dir = join(msa_embedding_dir, f"{target}_esm_attn.npz") #Full path
            esm_alledge_data = torch.tensor(np.load(esm_alledge_dir)["arr_0"]) #Load as 1X33X20XLXL matrix
            #trim off batch dim and start/end tokens
            esm_alledge_data = esm_alledge_data[0, ..., 1:-1, 1:-1]  #33X20XLxL
            #Flip dims
            esm_alledge_data = rearrange(esm_alledge_data, "d h w l -> w l d h") #depth(33) head(20) width length
            #Process copied from pair_missing process above
            esm_alledge_data = esm_alledge_data[mask_crop.bool()]  #Crop length
            esm_alledge_data = esm_alledge_data.permute(1, 0, 2, 3) #Flip length and width
            esm_alledge_data = esm_alledge_data[mask_crop.bool()] #Crop width
            esm_alledge_data = esm_alledge_data.permute(1, 0, 2, 3) #Flip lenght and width back
            data["esm_alledge"] = esm_alledge_data            

        if self.args.MT:
            c = sequence_output2.unsqueeze(0)
            #print(b.size())
            mask_c = c[mask_tmp.bool()]
            mask_c = mask_c.permute(1, 0, 2).unsqueeze(0)
            mask_c = mask_c[mask_tmp.bool()]
            mask_c = mask_c.permute(1, 0, 2)
            data['embed2'] = mask_c[aatype_start:aatype_start+torch.sum(res_dict["mask"]), aatype_start:aatype_start+torch.sum(res_dict["mask"]), ...]	 
        #==================================================
        if self.e2e:
            c = sequence_representation.unsqueeze(0)
            c = c.permute(0, 2, 1, 3)
            mask_c = c[mask_tmp.bool()]
            mask_c = mask_c.permute(1, 0, 2)

        data['single_representation'] = mask_a
        data['embed'] = mask_b
    
        data['coords'] = res_dict["coords"]
        data['mask'] = res_dict["mask"]
        data['record_lines'] = res_dict["record_lines"]
        '''all atom load'''
       
        prot_dict = {
            'aatype': np.asarray(prot.aatype),
            'all_atom_positions':np.asarray(prot.atom_positions),
            'all_atom_mask':np.asarray(prot.atom_mask),
        }						
        prot_dict = all_atom.make_atom14_positions(prot_dict)
        data['res_dict']['aatype'] = torch.tensor(np.asarray(prot.aatype))
        for key in prot_dict.keys():
            #print(f'{key}, ', prot_dict[key].shape)
            #data[key] = torch.tensor(prot_dict[key][aatype_start:aatype_start+torch.sum(res_dict["mask"]), ...])
            data[key] = torch.tensor(prot_dict[key][mask_pdb.bool()])
        
        protein_object = pr.from_pdb_string(pdb_str)
        residue_index = torch.tensor(protein_object.residue_index)
        protein_object = data_transforms.atom37_to_frames(
                                {'aatype': torch.tensor(protein_object.aatype),
                                'all_atom_positions': torch.tensor(protein_object.atom_positions),  # (..., 37, 3)
                                'all_atom_mask': torch.tensor(protein_object.atom_mask)})
        protein_object = data_transforms.atom37_to_torsion_angles(protein_object)
        protein_object.update({'residue_index': residue_index})
        protein_object.update({'residue_index': residue_index,
                                'chi_angles_sin_cos': protein_object["torsion_angles_sin_cos"][..., 3:, :],
                                'chi_mask': protein_object["torsion_angles_mask"][..., 3:],
                                'seq_mask':  torch.ones(protein_object["aatype"].shape, dtype=torch.float32)})
        for key in protein_object.keys():
            #print(f'{key}, ', protein_object[key].size())
            #data[key] = protein_object[key][aatype_start:aatype_start+torch.sum(res_dict["mask"]), ...]
            data[key] = protein_object[key][mask_pdb.bool()]

        data['seq_length'] = target_seq_len

        if self.distill:
            msa_embedding_dir = join(distill_dir, target)
            emd_dir = join(msa_embedding_dir, f'model_1.npz')
            emd = np.load(emd_dir)
            sequence_output = emd['pair']
            sequence_output = torch.tensor(sequence_output)
            b = sequence_output.unsqueeze(0)
            mask_b = b[mask_tmp.bool()]
            mask_b = mask_b.permute(1, 0, 2).unsqueeze(0)
            mask_b = mask_b[mask_tmp.bool()]
            mask_b = mask_b.permute(1, 0, 2)
            data['embed_distill'] = mask_b[aatype_start:aatype_start+torch.sum(res_dict["mask"]), aatype_start:aatype_start+torch.sum(res_dict["mask"]), ...]

            single = emd['single']
            single = torch.tensor(single)
            mask_a = a[mask_tmp.bool()]
            data['single_distill'] = mask_a[aatype_start:aatype_start+torch.sum(res_dict["mask"]), ...]

         
        if self.e2e:
            #print('here')
            if self.datatype == 'test':
                data['test'].update({'representation': sequence_representation.unsqueeze(0)})
            
            data['representation'] = mask_c[..., aatype_start:aatype_start+torch.sum(res_dict["mask"]), :]	
            #mas mask bert mask
            feature_dir = join(msa_embedding_dir, f'{target}/model_1_feature.pkl')
            features = None

            data['msa_mask'] = torch.ones(508, data['aatype'].shape[0])
            pseudo_beta, pseudo_beta_mask = pseudo_beta_fn(data['aatype'], data['all_atom_positions'], data['all_atom_mask'])
            data['pseudo_beta'] = pseudo_beta
            data['pseudo_beta_mask'] = pseudo_beta_mask

        if data['embed'].size(1) != data['aatype'].size(0):
            print("target is", target)
            print("embed_size is", data['embed'].size())   
            print("data size is", data['aatype'].size())
            print("start position is",start_position)
        data["target_seq"] = target_seq
        # print(target_seq)
        #print("THIS RETURN")
        return data
