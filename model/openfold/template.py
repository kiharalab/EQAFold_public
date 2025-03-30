# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import math
import torch
import torch.nn as nn

from model.openfold.primitives import Linear, Attention
from model.openfold.dropout import (
    DropoutRowwise,
    DropoutColumnwise,
)
from model.openfold.pair_transition import PairTransition
from model.openfold.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from model.openfold.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from model.openfold.checkpointing import checkpoint_blocks
from model.openfold.tensor_utils import (
    chunk_layer,
    permute_final_dims,
    flatten_final_dims,
)


class TemplatePointwiseAttention(nn.Module):
    """
    Implements Algorithm 17.
    """
    def __init__(self, c_t=64, c_z=128, c_hidden=16, no_heads=4, inf=1e5, **kwargs):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_z:
                Pair embedding channel dimension
            c_hidden:
                Hidden channel dimension
        """
        super(TemplatePointwiseAttention, self).__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf

        self.mha = Attention(
            self.c_z,
            self.c_t,
            self.c_t,
            self.c_hidden,
            self.no_heads,
            gating=False,
        )

    def forward(self, t, z, chunk_size, template_mask=None):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_t] pair embedding
            template_mask:
                [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """
        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])

        bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)

        # [*, N_res, N_res, N_temp, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))

        # [*, N_res, N_res, 1, C_z]
        mha_inputs = {
            "q_x": z,
            "k_x": t,
            "v_x": t,
            "biases": [bias],
        }
        if chunk_size is not None:
            z = chunk_layer(
                self.mha,
                mha_inputs,
                chunk_size=chunk_size,
                no_batch_dims=len(z.shape[:-2]),
            )
        else:
            z = self.mha(**mha_inputs)

        # [*, N_res, N_res, C_z]
        z = z.squeeze(-2)

        return z


class TemplatePairStackBlock(nn.Module):
    def __init__(
        self,
        c_t,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_heads,
        pair_transition_n,
        dropout_rate,
        inf,
        **kwargs,
    ):
        super(TemplatePairStackBlock, self).__init__()

        self.c_t = c_t
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.inf = inf

        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)

        self.tri_att_start = TriangleAttentionStartingNode(
            self.c_t,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            self.c_t,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            self.c_t,
            self.c_hidden_tri_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            self.c_t,
            self.c_hidden_tri_mul,
        )

        self.pair_transition = PairTransition(
            self.c_t,
            self.pair_transition_n,
        )

    def forward(self, z, mask, chunk_size, _mask_trans=True):
        single_templates = [
            t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)
        ]
        single_templates_masks = [
            m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)
        ]
        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]
            
            single = single + self.dropout_row(
                self.tri_att_start(
                    single,
                    chunk_size=chunk_size,
                    mask=single_mask
                )
            )
            single = single + self.dropout_col(
                self.tri_att_end(
                    single,
                    chunk_size=chunk_size,
                    mask=single_mask
                )
            )
            single = single + self.dropout_row(
                self.tri_mul_out(
                    single,
                    mask=single_mask
                )
            )
            single = single + self.dropout_row(
                self.tri_mul_in(
                    single,
                    mask=single_mask
                )
            )
            single = single + self.pair_transition(
                single,
                chunk_size=chunk_size,
                mask=single_mask if _mask_trans else None
            )

            single_templates[i] = single

        z = torch.cat(single_templates, dim=-4)

        return z


class TemplatePairStack(nn.Module):
    """
    Implements Algorithm 16.
    """
    def __init__(
        self,
        c_t=64,
        c_hidden_tri_att=16,
        c_hidden_tri_mul=64,
        no_blocks=2,
        no_heads=4,
        pair_transition_n=2,
        dropout_rate=0.25,
        blocks_per_ckpt=1,
        inf=1e9,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_att:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
        """
        super(TemplatePairStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = TemplatePairStackBlock(
                c_t=c_t,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                inf=inf,
            )
            self.blocks.append(block)

        self.layer_norm = nn.LayerNorm(c_t)

    def forward(
        self,
        t: torch.tensor,
        mask: torch.tensor,
        chunk_size: int,
        _mask_trans: bool = True,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if(mask.shape[-3] == 1):
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        (t,) = checkpoint_blocks(
            blocks=[
                partial(
                    b,
                    mask=mask,
                    chunk_size=chunk_size,
                    _mask_trans=_mask_trans,
                )
                for b in self.blocks
            ],
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        t = self.layer_norm(t)

        return t
