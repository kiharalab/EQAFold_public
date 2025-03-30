from os import rename
from model.ipa_openfold import *
from typing import Dict, Optional, Tuple

from model.openfold.loss import (sidechain_loss, 
                                torsion_angle_loss, 
                                compute_renamed_ground_truth, 
                                find_structural_violations, 
                                violation_loss,
                                supervised_chi_loss,
                                lddt_loss,
                                experimentally_resolved_loss,
                                distogram_loss,
                                backbone_loss,
                                masked_msa_loss)
from model.openfold.feats import atom14_to_atom37
from model.openfold.tensor_utils import tensor_tree_map
def fape_loss(
    outputs: Dict[str, torch.Tensor],
    batch_gt,
    batch_gt_frames
) -> torch.Tensor:
    #print(outputs["sm"]["frames"].size())
    bb_loss = backbone_loss(
        backbone_affine_tensor=batch_gt_frames["rigidgroups_gt_frames"][..., 0, :, :],
        backbone_affine_mask=batch_gt_frames['rigidgroups_gt_exists'][..., 0],
        traj=outputs["sm"]["frames"],
    )
    
    rename =compute_renamed_ground_truth(batch_gt, outputs['sm']['positions'][-1])
    sc_loss = sidechain_loss(
        sidechain_frames=outputs['sm']['sidechain_frames'],
        sidechain_atom_pos=outputs['sm']['positions'],
        rigidgroups_gt_frames=batch_gt_frames['rigidgroups_gt_frames'],
        rigidgroups_alt_gt_frames=batch_gt_frames['rigidgroups_alt_gt_frames'],
        rigidgroups_gt_exists=batch_gt_frames['rigidgroups_gt_exists'],
        renamed_atom14_gt_positions=rename['renamed_atom14_gt_positions'],
        renamed_atom14_gt_exists=rename['renamed_atom14_gt_exists'],
        alt_naming_is_better=rename['alt_naming_is_better'],
    )

    loss = 0.5 * bb_loss + 0.5 * sc_loss
    
    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss

class AlphaFoldLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, model):
        super(AlphaFoldLoss, self).__init__()
        self.model = model

    def forward(self, batch, batch_gt, batch_gt_frames, resolution):
        outputs = self.model(batch)
        msa_loss = 0
        msa_loss = masked_msa_loss(outputs["masked_msa_logits"], true_msa=batch['true_msa'][:, -1, ...].long(), bert_mask=batch['bert_mask'][:, -1, ...])
        fape = fape_loss(outputs, batch_gt, batch_gt_frames)
        angle_loss = supervised_chi_loss(outputs['sm']['angles'],
                                        outputs['sm']['unnormalized_angles'],
                                        aatype=batch_gt['aatype'],
                                        seq_mask=batch['seq_mask'][:, 0, ...],
                                        chi_mask=batch_gt_frames['chi_mask'],
                                        chi_angles_sin_cos=batch_gt_frames['chi_angles_sin_cos'],
                                        chi_weight=0.5,
                                        angle_norm_weight=0.01
                                        )
        dist_loss = distogram_loss(outputs['distogram_logits'],
                                        batch_gt['pseudo_beta'],
                                        batch_gt['pseudo_beta_mask'],)

        violation = find_structural_violations(batch_gt, outputs['sm']['positions'][-1],
                                                violation_tolerance_factor=12,
                                                clash_overlap_tolerance=1.5)
        violation_loss_ = violation_loss(violation, batch_gt['atom14_atom_exists'])
        vio_loss = torch.mean(violation_loss_)
        #print(violation_loss_)
        #fape += 1 * violation_loss_

        #lddt = self.plddt(outputs['single'])
        #final_position = atom14_to_atom37(outputs['sm']['positions'][-1], batch_gt) 
        plddt_loss = lddt_loss(outputs["lddt_logits"], outputs["final_atom_positions"], 
                                all_atom_positions=batch_gt['all_atom_positions'], 
                                all_atom_mask=batch_gt['all_atom_mask'],
                                resolution=resolution)
        #print(plddt_loss)
        exp_loss = experimentally_resolved_loss(outputs["experimentally_resolved_logits"], 
                                                atom37_atom_exists=batch_gt['atom37_atom_exists'],
                                                all_atom_mask=batch_gt['all_atom_mask'],
                                                resolution=resolution)
        loss = 0.5 * fape + 0.5 * angle_loss + 2 * msa_loss + 0.01 * plddt_loss + 0.3 * dist_loss
        return loss, fape, angle_loss, msa_loss, plddt_loss, dist_loss
