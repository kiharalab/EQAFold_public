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
                                backbone_loss)
from model.openfold.feats import atom14_to_atom37
from model.openfold.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
    ExtraMSAEmbedder,
)
#end to end
from .alphafold2 import *
from model.openfold.evoformer import EvoformerStack, ExtraMSAStack

import dataset.openfold_util.residue_constants as residue_constants

from model.openfold.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
)

from model.openfold.tensor_utils import (
    dict_multimap,
    tensor_tree_map,
)

from model.openfold.feats import (
    pseudo_beta_fn,
    build_extra_msa_feat,
    build_template_angle_feat,
    build_template_pair_feat,
    atom14_to_atom37,
)
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

class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    """

    def __init__(self, c_m=256, c_out=23, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
        """
        super(MaskedMSAHead, self).__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m, self.c_out, init="final")

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        # [*, N_seq, N_res, C_out]
        logits = self.linear(m)
        return logits

class AuxiliaryHeads(nn.Module):
    def __init__(self):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PerResidueLDDTCaPredictor()
        self.distogram = DistogramHead()
        self.masked_msa = MaskedMSAHead()
        self.experimentally_resolved = ExperimentallyResolvedHead()
        '''
        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )
        self.config = config
        '''
    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        #aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        aux_out["masked_msa_logits"] = masked_msa_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"]
        )
        aux_out[
            "experimentally_resolved_logits"
        ] = experimentally_resolved_logits
        '''
        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out["predicted_tm_score"] = compute_tm(
                tm_logits, **self.config.tm
            )
            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )
        '''
        return aux_out

class Alphafold(nn.Module):
    def __init__(self, args):
        super(Alphafold, self).__init__()
        #input embedders
        self.input_embedder = InputEmbedder()
        self.recycling_embedder = RecyclingEmbedder()
        self.template_angle_embedder = TemplateAngleEmbedder()
        self.template_pair_embedder = TemplatePairEmbedder()
        self.template_pair_stack = TemplatePairStack()
        self.template_pointwise_att = TemplatePointwiseAttention()
        self.extra_msa_embedder = ExtraMSAEmbedder()
        self.extra_msa_stack = ExtraMSAStack()
        self.evoformer = EvoformerStack(no_blocks=args.num_blocks)
        self.structure_module = StructureModule(trans_scale_factor=args.point_scale, no_blocks=args.ipa_depth)

        # self.plddt =  PerResidueLDDTCaPredictor()
        # self.experimentally_resolved = ExperimentallyResolvedHead()
        # self.distogram = DistogramHead()

        self.aux_heads = AuxiliaryHeads()

        self.args = args
        
        self.chunk_size = 4
        self._mask_trans = False
        self.embed_angles = True
        self.template_enabled = True
        self.extra_msa_enabled = True
        #self.train_evo = True
      
    def embed_templates(self, batch, z, pair_mask, templ_dim): 
        # Embed the templates one at a time (with a poor man's vmap)
        template_embeds = []
        n_templ = batch["template_aatype"].shape[templ_dim]
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx),
                batch,
            )

            single_template_embeds = {}
            if self.embed_angles:
                template_angle_feat = build_template_angle_feat(
                    single_template_feats,
                )

                # [*, S_t, N, C_m]
                a = self.template_angle_embedder(template_angle_feat)

                single_template_embeds["angle"] = a

            # [*, S_t, N, N, C_t]
            t = build_template_pair_feat(
                single_template_feats,
                inf=1e5,
                eps=1e-6
            )
            t = self.template_pair_embedder(t)

            single_template_embeds.update({"pair": t})

            template_embeds.append(single_template_embeds)

        template_embeds = dict_multimap(
            partial(torch.cat, dim=templ_dim),
            template_embeds,
        )

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            template_embeds["pair"], 
            pair_mask.unsqueeze(-3), 
            chunk_size=self.chunk_size,
            _mask_trans=self._mask_trans,
        )

        # [*, N, N, C_z]
        t = self.template_pointwise_att(
            t, 
            z, 
            template_mask=batch["template_mask"],
            chunk_size=self.chunk_size,
        )
        t = t * (torch.sum(batch["template_mask"]) > 0)

        ret = {}
        if self.embed_angles:
            ret["template_angle_embedding"] = template_embeds["angle"]

        ret.update({"template_pair_embedding": t})

        return ret

    def iteration(self, feats, m_1_prev, z_prev, x_prev, _recycle=True):
        # Primary output dictionary
        outputs = {}

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # Initialize the MSA and pair representations

        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        #with torch.set_grad_enabled(self.train_evo):
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
        )

        # Inject information from previous recycling iterations
        if _recycle:
            # Initialize the recycling embeddings, if needs be
            if None in [m_1_prev, z_prev, x_prev]:
                # [*, N, C_m]
                m_1_prev = m.new_zeros(
                    (*batch_dims, n, 256),
                )

                # [*, N, N, C_z]
                z_prev = z.new_zeros(
                    (*batch_dims, n, n, 128),
                )

                # [*, N, 3]
                x_prev = z.new_zeros(
                    (*batch_dims, n, residue_constants.atom_type_num, 3),
                )

            x_prev = pseudo_beta_fn(feats["aatype"], x_prev, None)
            #print(x_prev.size())
            # m_1_prev_emb: [*, N, C_m]
            # z_prev_emb: [*, N, N, C_z]
            m_1_prev_emb, z_prev_emb = self.recycling_embedder(
                m_1_prev,
                z_prev,
                x_prev,
            )

            # [*, S_c, N, C_m]
            m[..., 0, :, :] = m[..., 0, :, :] + m_1_prev_emb

            # [*, N, N, C_z]
            z = z + z_prev_emb

            # Possibly prevents memory fragmentation 
            del m_1_prev_emb, z_prev_emb

        # Embed the templates + merge with MSA/pair embeddings
        if self.template_enabled:
            template_mask = feats["template_mask"]
            if(torch.any(template_mask)):
                template_feats = {
                    k: v for k, v in feats.items() if k.startswith("template_")
                }
                template_embeds = self.embed_templates(
                    template_feats,
                    z,
                    pair_mask,
                    no_batch_dims,
                )

                # [*, N, N, C_z]
                z = z + template_embeds["template_pair_embedding"]

                if self.embed_angles:
                    # [*, S = S_c + S_t, N, C_m]
                    m = torch.cat(
                        [m, template_embeds["template_angle_embedding"]], 
                        dim=-3
                    )

                    # [*, S, N]
                    torsion_angles_mask = feats["template_torsion_angles_mask"]
                    msa_mask = torch.cat(
                        [feats["msa_mask"], torsion_angles_mask[..., 2]], 
                        dim=-2
                    )

        # Embed extra MSA features + merge with pairwise embeddings
        if self.extra_msa_enabled:
            # [*, S_e, N, C_e]
            a = self.extra_msa_embedder(build_extra_msa_feat(feats))

            # [*, N, N, C_z]
            z = self.extra_msa_stack(
                a,
                z,
                msa_mask=feats["extra_msa_mask"],
                chunk_size=self.chunk_size,
                pair_mask=pair_mask,
                _mask_trans=self._mask_trans,
            )

    # Run MSA + pair embeddings through the trunk of the network
    # m: [*, S, N, C_m]
    # z: [*, N, N, C_z]
    # s: [*, N, C_s]
    
        m, z, s = self.evoformer(
            m,
            z,
            msa_mask=msa_mask,
            pair_mask=pair_mask,
            chunk_size=self.chunk_size,
            _mask_trans=self._mask_trans,
        )
        # print(m.size())
        # print(z.size())
        # print(s.size())

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        # Predict 3D structure
        _, _, outputs["sm"] = self.structure_module(
            s,
            z,
            feats["aatype"],
            mask=feats["seq_mask"],
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [* N, N, C_z]
        z_prev = z

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]

        return outputs, m_1_prev, z_prev, x_prev

    def _disable_activation_checkpointing(self):
        self.template_pair_stack.blocks_per_ckpt = None
        self.evoformer.blocks_per_ckpt = None
        #self.extra_msa_stack.stack.blocks_per_ckpt = None

    def _enable_activation_checkpointing(self):
        self.template_pair_stack.blocks_per_ckpt = (
            1
        )
        self.evoformer.blocks_per_ckpt = (
            1
        )
        # self.extra_msa_stack.stack.blocks_per_ckpt = (
        #     1
        # )

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None

        # Disable activation checkpointing for the first few recycling iters
        is_grad_enabled = torch.is_grad_enabled()
        self._disable_activation_checkpointing()

        # Main recycling loop
        num_iters = batch["aatype"].shape[1]
        g = torch.Generator(device=torch.device('cpu'))
        def _randint(lower, upper):
                return int(torch.randint(
                        lower,
                        upper + 1,
                        (1,),
                        device=torch.device('cpu'),
                        generator=g,
                )[0])
        num_iters = _randint(1, 4)
        #num_iters = 4
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[:, cycle_no, ...]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                # Sidestep AMP bug (PyTorch issue #65766)
                if is_final_iter:
                    self._enable_activation_checkpointing()
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()
                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats,
                    m_1_prev,
                    z_prev,
                    x_prev,
                    _recycle=(num_iters > 1)
                )

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))
        return outputs

