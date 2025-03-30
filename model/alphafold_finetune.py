from os import rename
from model.ipa_openfold import *
from typing import Dict, Optional, Tuple

#JAKE
import model.graph as graph  
from torch_geometric.data import Data
from torch_geometric.nn.models import GAT
import model.egnn.egnn_clean as eg
from einops import rearrange
from model.openfold.loss import (sidechain_loss, 
                                torsion_angle_loss, 
                                compute_renamed_ground_truth, 
                                find_structural_violations, 
                                violation_loss,
                                supervised_chi_loss,
                                lddt_loss,
                                experimentally_resolved_loss,
                                distogram_loss,
                                compute_plddt)

from model.openfold.loss import model_lddt_error

from model.openfold.feats import atom14_to_atom37
from model.openfold.embedders import RecyclingEmbedder
#end to end
from .alphafold2 import *

from model.openfold.evoformer import EvoformerStack

from model.openfold.feats import (
    pseudo_beta_fn)
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

    #print(local_pred_pos.size())
    #print(local_target_pos.size())
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

def compute_fape2(
    pred_frames: T,
    target_frames: T,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
    hh_mask=None
) -> torch.Tensor:
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    #print(local_pred_pos.size())
    #print(local_target_pos.size())
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    #normed_error = normed_error * frames_mask[..., None]
    #normed_error = normed_error * positions_mask[..., None, :]
    normed_error = normed_error * hh_mask

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter
    # normed_error = torch.sum(normed_error, dim=-1)
    # normed_error = (
    #     normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    # )
    # normed_error = torch.sum(normed_error, dim=-1)
    # normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + torch.sum(hh_mask, dim=(-1, -2)))
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

def hhbond_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    hh_mask: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    renamed_gt_frames = (
        1.0 - alt_naming_is_better[..., None, None, None]
    ) * rigidgroups_gt_frames + alt_naming_is_better[
        ..., None, None, None
    ] * rigidgroups_alt_gt_frames
    
    # Steamroll the inputs
    sidechain_frames = sidechain_frames[-1]
    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = T.from_4x4(sidechain_frames)
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = T.from_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
        *batch_dims, -1, 3
    )
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)
    fape = compute_fape2(
        sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
        hh_mask=hh_mask,
    )

    return fape

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

class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s=384, c_out=37, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits

class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z=128, no_bins=64, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits
class BasicBlock2D(nn.Module):
    def __init__(self, channels=128, kernel_size=3, padding=1, dropout=0.1, stride=1, dilation=1):

        super(BasicBlock2D, self).__init__()

        padding = padding * dilation
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation)
        self.bn1 = nn.InstanceNorm2d(channels, affine=True)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride, padding, dilation)
        self.bn2 = nn.InstanceNorm2d(channels, affine=True)
        self.elu2 = nn.ELU()
        
    def forward(self, x):
        
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu1(out)

        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.elu2(out)

        return out

def embedding_loss(
    pred_emb: torch.Tensor,
    target_emb: torch.Tensor,
    seq_mask: torch.Tensor = None,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    
    **kwargs,
) -> torch.Tensor:
    mask = seq_mask[:, None, :] * seq_mask[:, :, None]
    error_dist = torch.sqrt(
        torch.sum((pred_emb - target_emb) ** 2, dim=-1) + eps
    )
    #print(error_dist.size())
    #error_dist = torch.clamp(error_dist, min=0, max=clamp_distance)
    #normed_error = error_dist / length_scale
    normed_error = error_dist * mask
    normed_error = torch.sum(normed_error, dim=(-1, -2, -3)) / (eps + torch.sum(mask, dim=(-1, -2, -3)))
    return normed_error

class AlphaFold(nn.Module):
    def __init__(self, args):
        super(AlphaFold, self).__init__()
        self.structure_module = StructureModule(trans_scale_factor=args.point_scale, no_blocks=args.ipa_depth, no_heads_ipa=12, c_ipa=16) #no_heads_ipa=24, c_ipa=64
        #JAKE
        #self.plddt =  PerResidueLDDTCaPredictor()

        self.mqa_transition_needed = False
        mqa_transition_layers = 384
        if args.esm_feats:
            #print("Add in ESM transition layers")
            self.mqa_transition_needed = True
            mqa_transition_layers = mqa_transition_layers + 33

        if args.rmsf_feats:
            #print("ADD in RMSF transition Layers")
            self.mqa_transition_needed = True
            mqa_transition_layers = mqa_transition_layers + 1

        if self.mqa_transition_needed:
            self.mqa_feat_transition = graph.mqa_feat_transition(in_features = mqa_transition_layers, out_features=384)
            self.plddt_feats = 384
        else:
            self.plddt_feats = 384

        self.edges_needed = False
        num_edge_feats = 0
        out_edge_feats = 128
        ### Turn off##
        if args.edge_feats:
            self.edges_needed = True
            num_edge_feats += 128
        if args.esm_edgefeats:
            self.edges_needed = True
            num_edge_feats += 33
        if args.esm_edgefeats_lastlayer:
            self.edges_needed = True
            num_edge_feats += 20
        if args.esm_edgefeats_alllayer:
            self.edges_needed = True
            num_edge_feats += 33 
            self.esm_groupconv = torch.nn.Conv2d(in_channels = 33 * 20, out_channels = 33, kernel_size = 21, groups = 33, padding='same')   

        if self.edges_needed:
            self.mqa_edgefeat_transition = graph.mqa_edgefeat_transition(in_channels = num_edge_feats, out_channels=out_edge_feats)
        else:  #If None of the above were added (No edge features are being used, just use edges as connections)
            num_edge_feats = 1
            out_edge_feats = 1
        ### End turn off ###


        assert args.graph_type in ["GAT", "GCN", "MLP", "EGNN"]
        if args.graph_type == "GCN":
            self.plddt =  graph.PerResidueLDDTGraphPred(in_channels = self.plddt_feats)
            print("WARNING args.graph_layers not being used!")
        elif args.graph_type == "GAT":
            print(f"Setting GAT with {args.graph_layers} layers")
            self.plddt = GAT(in_channels=self.plddt_feats, hidden_channels=128, num_layers=args.graph_layers, out_channels=50)
        elif args.graph_type == "MLP":
            print("WARNING args.graph_layers not being used!")
            self.plddt = PerResidueLDDTCaPredictor(c_in = self.plddt_feats)
        elif args.graph_type == "EGNN":
            self.plddt = eg.EGNN(in_node_nf=self.plddt_feats, hidden_nf=128, out_node_nf=50, in_edge_nf = out_edge_feats, n_layers=args.graph_layers)
        else:
            raise ValueError("This should never run!")
        

        self.contact_cutoff = args.contact_cutoff
        self.lddt_weight = args.lddt_weight
        self.lddt_weight_vector = args.lddt_weight_vector

        self.experimentally_resolved = ExperimentallyResolvedHead()
        # self.residual_blocks = nn.ModuleList([])
        # for i in range(5):
        #     self.residual_blocks.append(BasicBlock2D())
        if args.e2e:
            self.evoformer = EvoformerStack(no_blocks=args.num_blocks)
            self.residual_blocks = nn.ModuleList([])
            for i in range(5):
                self.residual_blocks.append(BasicBlock2D())
            self.distogram = DistogramHead()

        if args.distill:
            self.residual_blocks = nn.ModuleList([])
            for i in range(5):
                self.residual_blocks.append(BasicBlock2D())

        self.args = args

        if args.recycle:
            self.recycling_embedder = RecyclingEmbedder(c_m=384) if not args.e2e else RecyclingEmbedder(c_m=256)
        self.recycle = args.recycle
        self.MT = args.MT
        if self.MT:
            self.pair_project = nn.Linear(128 + 128, 128)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        #self.pair_project = nn.Linear(128 + 128, 128)
    def forward_training(self, embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution, representation=None, emb2=None, 
                        return_weighted_and_unweighted_plddt_loss = False):
        dist_out = None
        if self.recycle:
            output_bb, translation, outputs = self.forward_recycle(embedding, single_repr, aatype, batch_gt_frames, training=True, batch_gt=batch_gt, representation=representation)
            if self.args.e2e:
                distogram_logits = self.distogram(embedding)
                dist_out = distogram_loss(distogram_logits,
                                        batch_gt['pseudo_beta'],
                                        batch_gt['pseudo_beta_mask'],)
        else:
            if self.args.e2e:
                distogram_logits = self.distogram(embedding)
                dist_out = distogram_loss(distogram_logits,
                                        batch_gt['pseudo_beta'],
                                        batch_gt['pseudo_beta_mask'],)
            if self.args.distill:
                single_repr = batch_gt['single_distill']
            if self.MT:
                embedding = self.pair_project(torch.cat([embedding, emb2], dim=-1))
##################################################################CHANGES WITHIN HERE ###################################################3
            embedding = batch_gt['pair_missing']
            single = batch_gt['single_missing']

            output_bb, translation, outputs = self.structure_module(single, embedding, f=batch_gt['aatype_missing'], mask=batch_gt['seq_mask_missing'])

        #################### Do LDDT Prediction Before missing residue masking ##################################
        # print("batch_gt keys", batch_gt.keys())
        # for key in batch_gt.keys():
        #     if isinstance( batch_gt[key], torch.Tensor):
        #         print(key, batch_gt[key].shape)

        # Create Node Data  
        node_data = outputs["single"] 
        #print("Node data size is", node_data.shape)
        if self.args.esm_feats:
            node_data = torch.cat([node_data, batch_gt["esm"]], dim=2)
        #print("Node data size is", node_data.shape)  

        if self.args.rmsf_feats:
            node_data =  torch.cat([node_data,batch_gt["rmsf"]], dim=2)   
        #print("Node data size is", node_data.shape)

        if self.mqa_transition_needed:  #If any of the MQA feat transitions are being used push through the transition layer
            node_data = self.mqa_feat_transition(node_data)

        if self.args.graph_type in ["GAT", "GCN", "EGNN"]:
            nodes = graph.batch_nodes(node_data, feat_size = self.plddt_feats)
        
        #Make Edges
        if self.args.graph_type in ["GAT", "GCN", "EGNN"]:
            final_position_full = outputs['positions'][-1, :,:,:,:]  #BXLX16atomsX3coords - With missing residues
            #contact_mask = graph.create_full_contact_mask(batch_gt["seq_mask_missing"]) #B*L x B*L 
            edges = graph.contact_edges(final_position_full, contact=self.contact_cutoff, mask = None) # 2 X num_edges

        #Make edge features
        if self.edges_needed and (self.args.graph_type in ["GAT", "GCN", "EGNN"]):
            edge_data = torch.zeros(embedding.shape[0], embedding.shape[1],embedding.shape[2], 0, device=embedding.device, requires_grad=False)
            if self.args.edge_feats :
                edge_data = torch.cat([edge_data, embedding], dim=3)
            if self.args.esm_edgefeats:
                edge_data = torch.cat([edge_data, batch_gt["esm_edge"]], dim=3)
            if self.args.esm_edgefeats_lastlayer:
                edge_data = torch.cat([edge_data, batch_gt["esm_lastedge"]], dim=3)
            if self.args.esm_edgefeats_alllayer:
                #print("edge_data shape is ", edge_data.shape)
                attn = batch_gt["esm_alledge"]
                attn = self.esm_groupconv(attn) #Comes out as BX33XLXL
                attn = rearrange(attn, "b f l w -> b l w f")
                #print("final attn shape is ", attn.shape)
                edge_data = torch.cat([edge_data, attn], dim=3)

            #print("shape going into conv network is", edge_data.shape)
            edge_data = rearrange(edge_data, "b l w f -> b f l w")
            edge_data = self.mqa_edgefeat_transition(edge_data)   #THIS LINE CAUSES SEGFAULT WITH loss.backward() (fixed now)
            edge_data = rearrange(edge_data, "b f l w -> b l w f")
            #print("shape coming out of conv network is", edge_data.shape)

            blocked_edge_feats = graph.block_edge_feats(edge_data) # B*L X B*L X 128Feats
            edge_attr = graph.get_edge_attr(blocked_edge_feats, edges) # num_edges X 128Feats

        elif (self.args.graph_type in ["GAT", "GCN", "EGNN"]):
            edge_attr = torch.ones_like(edges[0]).reshape(-1, 1)  #Just have edge features of 1
        
        #Make coords
        coords = outputs['positions'][-1, :,:,1,:].reshape(-1, 3)    #B*LX3coords - With missing residues (index 1 is CA atoms)

        ## Prepare graph and clacluate graph lddt #########
        batch_size = outputs['single'].shape[0]   
        if self.args.graph_type in ["GAT", "GCN"]:
            #print("Outpust single is ", outputs['single'][0,0,:10])
            graphdata = Data(x=nodes, edge_index = edges) #torch_geometric.data.Data object
            lddt = self.plddt(graphdata.x, graphdata.edge_index) # for Graph network (Comes out as B*L x 50)
            lddt = lddt.reshape(batch_size, -1, 50) #Reshape to B x L x 50
            #print(lddt[0,0,:])
        elif self.args.graph_type == "EGNN":
            lddt, coords_out = self.plddt(nodes, coords, edges, edge_attr)
            lddt = lddt.reshape(batch_size, -1, 50) #Reshape to B x L x 50

        elif self.args.graph_type == "MLP":
            lddt = self.plddt(node_data) # original Network #self.plddt(outputs['single']) # original Network
        else:
            print("This should never run!")
            raise ValueError
        # print("finished running model at", str(datetime.datetime.now()))
        # print("lddt shape is", lddt.shape)  #B X L X 50
        # print("exiting")
        # exit()

        #Remove missing residues(Follows same process as below)
        max_len = min(batch_gt['single_missing'].size(1), aatype.size(1))
        s = torch.zeros_like(lddt)
        s = s[:, :max_len, ...]  #Change L to max_len
        for b in range(lddt.size(0)): #For each element in the batch
            pad = lddt[b] #Full batch (LX50) of this batch
            mask_ = batch_gt['missing'][b]  #Get mask out all missing residues (unresolved and padded) in this batch element
            mask_b = pad[mask_.bool()]      #Mask out the unresolved / padded residue from this batch element
            s[b, :mask_b.size(0), :] = mask_b  #Set the masked version in the output
        lddt = s
        #print("lddt shape is now", lddt.shape)  #B X max_len X 50
        lddt_values = compute_plddt(lddt)  #B X max_len (Converts binnded predictions to 0-100 LDDT values)


        # END LDDT Prediction #####################################################################################
        pred_frames = torch.stack(output_bb)
        #print('frames size before: ', pred_frames.size())
        #This has to be the code to covnert full-length to resolved AA length - jake
        #missing residue update
        def mask_multi(emb, mask):
            #print(emb.size())
            batch_dims = torch.arange(len(emb.size()))[2:]
            emb = emb.permute(1, 0, *batch_dims)
            mask_b = emb[mask.bool()]
            return mask_b.permute(1, 0, *batch_dims)
    
        #print(aatype.size())
        #print(batch_gt['single_missing'].size(1))
        max_len = min(batch_gt['single_missing'].size(1), aatype.size(1))
        for key in outputs.keys():
            if key == 'single':
                tensor = outputs[key]
                s = torch.zeros_like(tensor)
                s = s[:, :max_len, ...]
                for b in range(tensor.size(0)):
                    pad = tensor[b]
                    mask_ = batch_gt['missing'][b]
                    mask_b = pad[mask_.bool()]
                    s[b, :mask_b.size(0), ...] = mask_b
                outputs[key] = s
                continue
            batch_dims = torch.arange(len(outputs[key].size()))[2:]
            #print(batch_dims)
            tensor = outputs[key].permute(1, 0, *batch_dims)
            s = torch.zeros_like(tensor)
            s = s[:, :, :max_len, ...]
            for b in range(tensor.size(0)):
                pad = tensor[b]
                mask_ = batch_gt['missing'][b]
                mask_tensor = mask_multi(pad, mask_)
                #print(mask_tensor.size())
                l = mask_tensor.size(1)
                #print(mask_tensor.size())
                #print(s.size())
                s[b, :, :l, ...] = mask_tensor
                #print(e.size())
            outputs[key] = s.permute(1, 0, *batch_dims)
        pred_frames = outputs['frames']

##################################################################CHANGES WITHIN HERE ###################################################3
        bb_loss = backbone_loss(
            backbone_affine_tensor=batch_gt_frames["rigidgroups_gt_frames"][..., 0, :, :],
            backbone_affine_mask=batch_gt_frames['rigidgroups_gt_exists'][..., 0],
            traj=pred_frames,
        )

        #the sidechain 
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
        fape = 0.5 * bb_loss + 0.5 * sc_loss
        
        #print('fape: ', fape.size())
        vio_loss = 0
        plddt_loss = 0
        #if not self.training: #IF BLOCK REMOVED BY JAKE
        batch_gt.update({'aatype': aatype})
        violation = find_structural_violations(batch_gt, outputs['positions'][-1],
                                            violation_tolerance_factor=12,
                                            clash_overlap_tolerance=1.5)
        violation_loss_ = violation_loss(violation, batch_gt['atom14_atom_exists'])
        vio_loss = torch.mean(violation_loss_)
        #print(violation_loss_)
        fape += 1 * violation_loss_

        #print(plddt_loss)
        experimentally_resolved_logits = self.experimentally_resolved(outputs['single'])
        exp_loss = experimentally_resolved_loss(experimentally_resolved_logits, 
                                                atom37_atom_exists=batch_gt['atom37_atom_exists'],
                                                all_atom_mask=batch_gt['all_atom_mask'],
                                                resolution=resolution)
        fape += 0.01 * exp_loss

        final_position = atom14_to_atom37(outputs['positions'][-1], batch_gt) 


        #Writing out so there's some clarity in this code. 
        #If aa-lddt is set, we DONT use just CA atoms, we use all atoms
        if self.args.aa_lddt:  
            ca_lddt = False
        else:
            ca_lddt = True #Default

        plddt_loss = lddt_loss(lddt, final_position, 
                                all_atom_positions=batch_gt['all_atom_positions'], 
                                all_atom_mask=batch_gt['all_atom_mask'],
                                resolution=resolution, 
                                weight_vec = self.lddt_weight_vector, 
                                ca_lddt = ca_lddt) #Openfold Loss calculation
        

        plddt_loss_unweighted = lddt_loss(lddt, final_position, 
                                          all_atom_positions=batch_gt['all_atom_positions'], 
                                          all_atom_mask=batch_gt['all_atom_mask'],
                                          resolution=resolution, 
                                          weight_vec = torch.ones(50).to(lddt.device), 
                                          ca_lddt = ca_lddt) #Openfold Loss calculation
        
        if return_weighted_and_unweighted_plddt_loss:
            lddt_error = model_lddt_error(lddt, 
                                        final_position, 
                                        all_atom_positions=batch_gt['all_atom_positions'], 
                                        all_atom_mask=batch_gt['all_atom_mask'])

        
        fape += (self.lddt_weight) * plddt_loss
        fape = torch.mean(fape)
        #print('fape before:', fape.item())
        fape += 0.5 * angle_loss
        #print('fape after:', fape.item())
        if self.args.e2e:
            fape += 0.3 * dist_out
            #print(dist_out)
        #print(translation.size())
        hh_loss, hydro_loss, ionic_loss = None, None, None
        if self.args.hhbond:
            hh_loss = hhbond_loss(
                sidechain_frames=outputs['sidechain_frames'],
                sidechain_atom_pos=outputs['positions'],
                rigidgroups_gt_frames=batch_gt_frames['rigidgroups_gt_frames'],
                rigidgroups_alt_gt_frames=batch_gt_frames['rigidgroups_alt_gt_frames'],
                rigidgroups_gt_exists=batch_gt_frames['rigidgroups_gt_exists'],
                renamed_atom14_gt_positions=rename['renamed_atom14_gt_positions'],
                renamed_atom14_gt_exists=rename['renamed_atom14_gt_exists'],
                alt_naming_is_better=rename['alt_naming_is_better'],
                hh_mask=batch_gt['hhbond'],
            )
            hydro_loss = hhbond_loss(
                sidechain_frames=outputs['sidechain_frames'],
                sidechain_atom_pos=outputs['positions'],
                rigidgroups_gt_frames=batch_gt_frames['rigidgroups_gt_frames'],
                rigidgroups_alt_gt_frames=batch_gt_frames['rigidgroups_alt_gt_frames'],
                rigidgroups_gt_exists=batch_gt_frames['rigidgroups_gt_exists'],
                renamed_atom14_gt_positions=rename['renamed_atom14_gt_positions'],
                renamed_atom14_gt_exists=rename['renamed_atom14_gt_exists'],
                alt_naming_is_better=rename['alt_naming_is_better'],
                hh_mask=batch_gt['hydro'],
            )
            ionic_loss = hhbond_loss(
                sidechain_frames=outputs['sidechain_frames'],
                sidechain_atom_pos=outputs['positions'],
                rigidgroups_gt_frames=batch_gt_frames['rigidgroups_gt_frames'],
                rigidgroups_alt_gt_frames=batch_gt_frames['rigidgroups_alt_gt_frames'],
                rigidgroups_gt_exists=batch_gt_frames['rigidgroups_gt_exists'],
                renamed_atom14_gt_positions=rename['renamed_atom14_gt_positions'],
                renamed_atom14_gt_exists=rename['renamed_atom14_gt_exists'],
                alt_naming_is_better=rename['alt_naming_is_better'],
                hh_mask=batch_gt['ionic'],
            )

            hh_loss = torch.mean(hh_loss)
            hydro_loss = torch.mean(hydro_loss)
            ionic_loss = torch.mean(ionic_loss)
            fape += 0.01 * (ionic_loss + hh_loss + hydro_loss)

        seq_len = torch.mean(batch_gt["seq_length"].float())
        crop_len = torch.tensor(aatype.shape[-1]).to(device=aatype.device)

        fape = fape * torch.sqrt(min(seq_len, crop_len))
        #print(fape)


        if return_weighted_and_unweighted_plddt_loss:  #Package the 2 plddt losses (and lddt values) into 1 when I need it
            return translation*self.args.point_scale, fape, outputs['positions'], vio_loss, angle_loss, (plddt_loss, plddt_loss_unweighted, lddt_values, lddt_error), dist_out, \
                torch.mean(bb_loss), torch.mean(sc_loss), hh_loss, hydro_loss, ionic_loss
        else: #Return as normal
            return translation*self.args.point_scale, fape, outputs['positions'], vio_loss, angle_loss, plddt_loss, dist_out, \
                torch.mean(bb_loss), torch.mean(sc_loss), hh_loss, hydro_loss, ionic_loss


    def forward_testing(self, embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution, representation=None, emb2=None, 
                        return_weighted_and_unweighted_plddt_loss = False):

        embedding = batch_gt['pair_missing']
        single = batch_gt['single_missing']
        output_bb, translation, outputs = self.structure_module(single, embedding, f=batch_gt['aatype_missing'], mask=batch_gt['seq_mask_missing'])

        # Create Node Data  
        node_data = outputs["single"] 
        if self.args.esm_feats:
            node_data = torch.cat([node_data, batch_gt["esm"]], dim=2) 
        if self.args.rmsf_feats:
            node_data =  torch.cat([node_data,batch_gt["rmsf"]], dim=2)   
        if self.mqa_transition_needed:  #If any of the MQA feat transitions are being used push through the transition layer
            node_data = self.mqa_feat_transition(node_data)
        if self.args.graph_type in ["GAT", "GCN", "EGNN"]:
            nodes = graph.batch_nodes(node_data, feat_size = self.plddt_feats)
        
        #Make Edges
        if self.args.graph_type in ["GAT", "GCN", "EGNN"]:
            final_position_full = outputs['positions'][-1, :,:,:,:]  #BXLX16atomsX3coords - With missing residues
            edges = graph.contact_edges(final_position_full, contact=self.contact_cutoff, mask = None) # 2 X num_edges

        #Make edge features
        if self.edges_needed and (self.args.graph_type in ["GAT", "GCN", "EGNN"]):
            edge_data = torch.zeros(embedding.shape[0], embedding.shape[1],embedding.shape[2], 0, device=embedding.device, requires_grad=False)
            if self.args.edge_feats :
                edge_data = torch.cat([edge_data, embedding], dim=3)
            if self.args.esm_edgefeats:
                edge_data = torch.cat([edge_data, batch_gt["esm_edge"]], dim=3)
            if self.args.esm_edgefeats_lastlayer:
                edge_data = torch.cat([edge_data, batch_gt["esm_lastedge"]], dim=3)
            if self.args.esm_edgefeats_alllayer:
                #print("edge_data shape is ", edge_data.shape)
                attn = batch_gt["esm_alledge"]
                attn = self.esm_groupconv(attn) #Comes out as BX33XLXL
                attn = rearrange(attn, "b f l w -> b l w f")
                #print("final attn shape is ", attn.shape)
                edge_data = torch.cat([edge_data, attn], dim=3)

            edge_data = rearrange(edge_data, "b l w f -> b f l w")
            edge_data = self.mqa_edgefeat_transition(edge_data)   #THIS LINE CAUSES SEGFAULT WITH loss.backward() (fixed now)
            edge_data = rearrange(edge_data, "b f l w -> b l w f")
            #print("shape coming out of conv network is", edge_data.shape)

            blocked_edge_feats = graph.block_edge_feats(edge_data) # B*L X B*L X 128Feats
            edge_attr = graph.get_edge_attr(blocked_edge_feats, edges) # num_edges X 128Feats

        elif (self.args.graph_type in ["GAT", "GCN", "EGNN"]):
            edge_attr = torch.ones_like(edges[0]).reshape(-1, 1)  #Just have edge features of 1
        
        #Make coords
        coords = outputs['positions'][-1, :,:,1,:].reshape(-1, 3)    #B*LX3coords - With missing residues (index 1 is CA atoms)

        ## Prepare graph and clacluate graph lddt #########
        batch_size = outputs['single'].shape[0]   
        if self.args.graph_type in ["GAT", "GCN"]:
            #print("Outpust single is ", outputs['single'][0,0,:10])
            graphdata = Data(x=nodes, edge_index = edges) #torch_geometric.data.Data object
            lddt = self.plddt(graphdata.x, graphdata.edge_index) # for Graph network (Comes out as B*L x 50)
            lddt = lddt.reshape(batch_size, -1, 50) #Reshape to B x L x 50
            #print(lddt[0,0,:])
        elif self.args.graph_type == "EGNN":
            lddt, coords_out = self.plddt(nodes, coords, edges, edge_attr)
            lddt = lddt.reshape(batch_size, -1, 50) #Reshape to B x L x 50

        elif self.args.graph_type == "MLP":
            lddt = self.plddt(node_data) # original Network #self.plddt(outputs['single']) # original Network
        else:
            print("This should never run!")
            raise ValueError

        #print("lddt shape is now", lddt.shape)  #B X max_len X 50
        lddt_values = compute_plddt(lddt)  #B X max_len (Converts binnded predictions to 0-100 LDDT values)
        return outputs['positions'], lddt_values



    def forward_recycle(self, embedding, single_repr, aatype, batch_gt_frames=None, training=True, num_iterations=4, 
                        batch_gt=None, protein_test=None, representation=None):
        with torch.no_grad():
            if training:
                if self.args.e2e:
                    msa_mask, pair_mask = None, None
                    seq_mask = batch_gt_frames["seq_mask"]
                    pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
                    msa_mask = batch_gt["msa_mask"]
                    representation, embedding, single_repr = self.evoformer(
                        representation,
                        embedding,
                        msa_mask=msa_mask,
                        pair_mask=pair_mask,
                        chunk_size=4,
                        _mask_trans=False,
                    )
                output_bb, translation, outputs = self.structure_module(single_repr, embedding, f=aatype, mask=batch_gt_frames['seq_mask'])
            else:
                if self.args.e2e:
                    msa_mask, pair_mask = None, None
                    representation, embedding, single_repr = self.evoformer(
                        representation,
                        embedding,
                        msa_mask=msa_mask,
                        pair_mask=pair_mask,
                        chunk_size=4,
                        _mask_trans=False,
                    )
                output_bb, translation, outputs = self.structure_module(single_repr, embedding, f=aatype, mask=None)
            for i in range(num_iterations-2):
                if training:
                    final_pos = atom14_to_atom37(outputs['positions'][-1], batch_gt)
                else:
                    final_pos = atom14_to_atom37(outputs['positions'][-1], protein_test)
                x_prev = pseudo_beta_fn(aatype, final_pos, None)
                if not self.args.e2e:
                    m_1_prev_emb, z_prev_emb = self.recycling_embedder(
                        outputs["single"],
                        embedding,
                        x_prev,
                    )
                    single_repr += m_1_prev_emb
                else:
                     m_1_prev_emb, z_prev_emb = self.recycling_embedder(
                        representation[:, 0, :, :],
                        embedding,
                        x_prev,
                    )
                embedding += z_prev_emb
                if not self.args.e2e:
                    del m_1_prev_emb, z_prev_emb
                if training:
                    if self.args.e2e:
                        msa_mask, pair_mask = None, None
                        seq_mask = batch_gt_frames["seq_mask"]
                        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
                        msa_mask = batch_gt["msa_mask"]
                        representation[:, 0, :, :] += m_1_prev_emb
                        del m_1_prev_emb, z_prev_emb
                        representation, embedding, single_repr = self.evoformer(
                            representation,
                            embedding,
                            msa_mask=msa_mask,
                            pair_mask=pair_mask,
                            chunk_size=4,
                            _mask_trans=False,
                        )
                    output_bb, translation, outputs = self.structure_module(single_repr, embedding, f=aatype, mask=batch_gt_frames['seq_mask'])
                else:
                    if self.args.e2e:
                        msa_mask, pair_mask = None, None
                        representation[:, 0, :, :] += m_1_prev_emb
                        representation, embedding, single_repr = self.evoformer(
                            representation,
                            embedding,
                            msa_mask=msa_mask,
                            pair_mask=pair_mask,
                            chunk_size=4,
                            _mask_trans=False,
                        )
                    output_bb, translation, outputs = self.structure_module(single_repr, embedding, f=aatype, mask=None)

        if training:
            final_pos = atom14_to_atom37(outputs['positions'][-1], batch_gt)
        else:
            final_pos = atom14_to_atom37(outputs['positions'][-1], protein_test)
        x_prev = pseudo_beta_fn(aatype, final_pos, None)
        if not self.args.e2e:
            m_1_prev_emb, z_prev_emb = self.recycling_embedder(
                outputs["single"],
                embedding,
                x_prev,
            )
            single_repr += m_1_prev_emb
        else:
            m_1_prev_emb, z_prev_emb = self.recycling_embedder(
                representation[:, 0, :, :],
                embedding,
                x_prev,
            )
        #training issue
        embedding_new = embedding + z_prev_emb
        if not self.args.e2e:
            del m_1_prev_emb, z_prev_emb
        if training:
            if self.args.e2e:
                msa_mask, pair_mask = None, None
                seq_mask = batch_gt_frames["seq_mask"]
                pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
                msa_mask = batch_gt["msa_mask"]
                #representation[:, 0, :, :] += m_1_prev_emb
                #del m_1_prev_emb, z_prev_emb
                representation, embedding_new, single_repr = self.evoformer(
                    representation,
                    embedding_new,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                    chunk_size=4,
                    _mask_trans=False,
                )
            output_bb, translation, outputs = self.structure_module(single_repr, embedding_new, f=aatype, mask=batch_gt_frames['seq_mask'])
        else:
            if self.args.e2e:
                msa_mask, pair_mask = None, None
                #representation[:, 0, :, :] += m_1_prev_emb
                representation, embedding_new, single_repr = self.evoformer(
                    representation,
                    embedding_new,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,
                    chunk_size=4,
                    _mask_trans=False,
                )
            output_bb, translation, outputs = self.structure_module(single_repr, embedding_new, f=aatype, mask=None)
        return output_bb, translation, outputs
       
    def forward(self, embedding, single_repr, aatype, batch_gt, batch_gt_frames, 
                resolution, representation=None, training=True, protein_test=None, emb2=None, return_weighted_and_unweighted_plddt_loss = False):

        if self.args.test_only:
            return self.forward_testing(embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution, representation=representation,
                                        return_weighted_and_unweighted_plddt_loss = return_weighted_and_unweighted_plddt_loss)
        
        else:
            return self.forward_training(embedding, single_repr, aatype, batch_gt, batch_gt_frames, resolution, representation=representation,
                                            return_weighted_and_unweighted_plddt_loss = return_weighted_and_unweighted_plddt_loss)
