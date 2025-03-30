from os import rename
from model.ipa_openfold import *
from typing import Dict, Optional, Tuple

from model.openfold.loss import (sidechain_loss, 
                                torsion_angle_loss, 
                                compute_renamed_ground_truth, 
                                find_structural_violations, 
                                violation_loss,
                                supervised_chi_loss,
                                lddt_loss)
from model.openfold.feats import atom14_to_atom37
from model.openfold.embedders import RecyclingEmbedder
#end to end
from .alphafold2 import *

def compute_fape(
    pred_frames: T,
    target_frames: T,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    backbone_affine_tensor: torch.Tensor,
    backbone_affine_mask: torch.Tensor,
    traj: torch.Tensor,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    pred_aff = T.from_tensor(traj)
    gt_aff = T.from_tensor(backbone_affine_tensor)

    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        backbone_affine_mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        backbone_affine_mask[None],
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            backbone_affine_mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            backbone_affine_mask[None],
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss

class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins=50, c_in=384, c_hidden=128):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = nn.LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s

class DistPredictor(nn.Module):
    def __init__(self, args, channels=128):
        super(DistPredictor, self).__init__()
        out_channels_dist = args.out_channels_dist
        self.out_channels_angle = args.out_channels_angle
        out_channels_mu = args.out_channels_mu
        out_channels_theta = args.out_channels_theta
        out_channels_sce = args.out_channels_sce
        out_channels_no = args.out_channels_no

        self.pair_embed = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
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

    def forward(self, embedding):
        embedding = embedding.permute(0, 3, 1, 2)
        outs = embedding + self.pair_embed(embedding)
        pair = outs.permute(0, 2, 3, 1)
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
        output = [dist_outs, mu_outs, theta_outs, rho_outs, sce_outs, no_outs, phi_outs, psi_outs]

        return output, pair

class AttentiveDist6(nn.Module):
    def __init__(self, args):
        super(AttentiveDist6, self).__init__()
        self.ipa = StructureModule(trans_scale_factor=args.point_scale, no_blocks=args.ipa_depth)
        self.plddt =  PerResidueLDDTCaPredictor()
        if args.e2e:
            self.dist = DistPredictor(args)
        self.args = args

        if args.recycle:
            self.recycling_embedder = RecyclingEmbedder()
        self.recycle = args.recycle
        # self.sinlge_project = nn.Linear(768, 384)
        # self.pair_project = nn.Linear(144, 128)
        
        '''
        self.resnet = Evoformer(
				dim = 64,
				depth = 48,
				seq_len = 1024,
				heads = 8,
				dim_head = 32,
				attn_dropout = 0,
				ff_dropout = 0.2
        )
        '''
        
    def forward_training(self, embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution):
        #print(mask.size())
        #print('embedding: ', single_repr.size())
        #print('aatype: ', aatype.size())
        dist_out = None
        #batch, length, hidden = single_repr.size()
        # psudo_pair = torch.ones(batch, length, length, 64).cuda(1)
        # psudo_msa = torch.ones(batch, 384, length, 64).cuda(1)
        # with torch.no_grad():
        #     x, m = self.resnet(psudo_pair ,psudo_msa)

        if self.args.e2e:
            dist_out, embedding = self.dist(embedding)
        #for msa transformer embedding
        # embedding = self.pair_project(embedding)
        # single_repr = self.sinlge_project(single_repr)
        if self.recycle:
            output_bb, translation, outputs = self.forward_recycle(embedding, single_repr, aatype, batch_gt_frames, training=True)
        else:
            output_bb, translation, outputs = self.ipa_module(single_repr, embedding, f=aatype, mask=batch_gt_frames['seq_mask'])
        #output_bb, translation, outputs = self.ipa(single_repr, embedding, f=aatype, mask=None)
       
        pred_frames = torch.stack(output_bb)
        #print(pred_frames.size())

        #target_frames = T(T_true[0],T_true[1]).to_4x4()
        #target_frames = target_frames.repeat(8, 1, 1 ,1)
        #print(target_frames.size())
        bb_loss = backbone_loss(
            backbone_affine_tensor=batch_gt_frames["rigidgroups_gt_frames"][..., 0, :, :],
            backbone_affine_mask=batch_gt_frames['rigidgroups_gt_exists'][..., 0],
            traj=pred_frames,
        )

        #the fucking sidechain 
        rename =compute_renamed_ground_truth(batch_gt, outputs['positions'][-1])
       
        sc_loss = sidechain_loss(
            sidechain_frames=outputs['sidechain_frames'],
            sidechain_atom_pos=outputs['positions'],
            rigidgroups_gt_frames=batch_gt_frames['rigidgroups_gt_frames'],
            rigidgroups_alt_gt_frames=batch_gt_frames['rigidgroups_alt_gt_frames'],
            rigidgroups_gt_exists=batch_gt_frames['rigidgroups_gt_exists'],
            renamed_atom14_gt_positions=rename['renamed_atom14_gt_positions'],
            renamed_atom14_gt_exists=rename['renamed_atom14_gt_exists'],
            alt_naming_is_better=rename['alt_naming_is_better'],
        )
        
        angle_loss = supervised_chi_loss(outputs['angles'],
                                        outputs['unnormalized_angles'],
                                        aatype=aatype,
                                        seq_mask=batch_gt_frames['seq_mask'],
                                        chi_mask=batch_gt_frames['chi_mask'],
                                        chi_angles_sin_cos=batch_gt_frames['chi_angles_sin_cos'],
                                        chi_weight=0.5,
                                        angle_norm_weight=0.01
                                        )
        #print(angle_loss)
        plddt_loss = 0
        lddt = self.plddt(outputs['single'])
        final_position = atom14_to_atom37(outputs['positions'][-1], batch_gt) 
        plddt_loss = lddt_loss(lddt, final_position, 
                                all_atom_positions=batch_gt['all_atom_positions'], 
                                all_atom_mask=batch_gt['all_atom_mask'],
                                resolution=resolution)
        #print(plddt_loss)
        fape = 0.5 * bb_loss + 0.5 * sc_loss
        vio_loss = 0
        if not self.training:
            batch_gt.update({'aatype': aatype})
            violation = find_structural_violations(batch_gt, outputs['positions'][-1],
                                                violation_tolerance_factor=12,
                                                clash_overlap_tolerance=1.5)
            violation_loss_ = violation_loss(violation, batch_gt['atom14_atom_exists'])
            vio_loss = torch.mean(violation_loss_)
            #print(violation_loss_)
            fape = 0.5 * bb_loss + 0.5 * sc_loss + 1 * violation_loss_
        fape = torch.mean(fape)
        fape += 0.5 * angle_loss + 0.01 * plddt_loss
        #print(translation.size())
        return translation*self.args.point_scale, fape, outputs['positions'], vio_loss, angle_loss, plddt_loss, dist_out

    def forward_testing(self, embedding, single_repr, aatype):
        dist_out = None
        if self.args.e2e:
            dist_out, embedding = self.dist(embedding)
        if self.recycle:
            output_bb, translation, outputs = self.forward_recycle(embedding, single_repr, aatype, training=False)
        else:
            output_bb, translation, outputs = self.ipa(single_repr, embedding, f=aatype, mask=None)
       
        pred_frames = torch.stack(output_bb)
        return translation*self.args.point_scale, outputs['positions']

    def forward_recycle(self, embedding, single_repr, aatype, batch_gt_frames=None, training=True, num_iterations=3):
        # batch, length, hidden = single_repr.size()
        # m_1_prev = torch.ones(batch, length, 384).cuda(1)
        # # [*, N, N, C_z]
        # z_prev = torch.ones(batch, length, length, 128).cuda(1)
        # # [*, N, 3]
        # x_prev = torch.ones(batch, length, 3).cuda(1)
        with torch.no_grad():
            for i in range(num_iterations-1):
                if training:
                    output_bb, translation, outputs = self.ipa(single_repr, embedding, f=aatype, mask=batch_gt_frames['seq_mask'])
                else:
                    output_bb, translation, outputs = self.ipa(single_repr, embedding, f=aatype, mask=None)
                m_1_prev_emb, z_prev_emb = self.recycling_embedder(
                    outputs["single"],
                    embedding,
                    translation*self.args.point_scale,
                )
                single_repr += m_1_prev_emb
                embedding += z_prev_emb
                del m_1_prev_emb, z_prev_emb
        if training:
            output_bb, translation, outputs = self.ipa(single_repr, embedding, f=aatype, mask=batch_gt_frames['seq_mask'])
        else:
            output_bb, translation, outputs = self.ipa(single_repr, embedding, f=aatype, mask=None)
        return output_bb, translation, outputs
       
    def forward(self, embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution, training=True):
        if training:
            return self.forward_training(embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution)
        else:
            return self.forward_testing(embedding, single_repr, aatype)
