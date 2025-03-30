import torch
from torch import nn, einsum
from inspect import isfunction
from functools import partial
from itertools import islice, cycle
from collections import namedtuple
import torch.nn.functional as F

from math import sqrt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
# constants

Logits = namedtuple('Logits', ['distance', 'theta', 'phi', 'omega'])

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

# helper classes

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return self.val

# feed forward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        gating = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, attn_bias = None, context = None, context_mask = None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # query / key similarities

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # add attention bias, if supplied (for pairwise to msa attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention
        #print('before softmax: ', dots.size())
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        gates = self.gating(x)
        out = out * gates            

        # combine to out

        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        sparse_attn = False,
        row_attn = True,
        col_attn = True,
        accept_edges = False,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        attn_class = SparseAttention if sparse_attn else Attention

        self.norm = nn.LayerNorm(dim)

        self.attn_width = attn_class(dim = dim, heads = heads, **kwargs)
        self.attn_height = attn_class(dim = dim, heads = heads, **kwargs)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

        self.row_attn = row_attn
        self.col_attn = col_attn

    def forward(self, x, edges = None, mask = None):
        b, h, w, d = x.shape

        x = self.norm(x)

        # axial mask

        w_mask = h_mask = None

        if exists(mask):
            w_mask = rearrange(mask, 'b h w -> (b w) h')
            h_mask = rearrange(mask, 'b h w -> (b h) w')

        # calculate attention bias

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)

        # axial attention

        out = 0
        axial_attn_count = 0

        if self.row_attn:
            w_x = rearrange(x, 'b h w d -> (b w) h d')
            if exists(attn_bias):
                attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = w)

            w_out = self.attn_width(w_x, mask = w_mask, attn_bias = attn_bias)
            w_out = rearrange(w_out, '(b w) h d -> b h w d', h = h, w = w)

            out += w_out
            axial_attn_count += 1

        if self.col_attn:
            h_x = rearrange(x, 'b h w d -> (b h) w d')
            if exists(attn_bias):
                attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = h)

            h_out = self.attn_height(h_x, mask = h_mask, attn_bias = attn_bias)
            h_out = rearrange(h_out, '(b h) w d -> b h w d', h = h, w = w)

            out += h_out
            axial_attn_count += 1

        out /= axial_attn_count

        return out

class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        #hidden_dim = None,
        hidden_dim = 128,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'ingoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'outgoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * left_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

# evoformer blocks
class OuterMean(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = None
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') + rearrange(right, 'b m j d -> b m () j d')
        outer = outer.mean(dim = 1)
        return self.proj_out(outer)

class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)

        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True)
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None
    ):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr)

        x = self.triangle_attention_ingoing(x, edges = x, mask = mask) + x
        x = self.triangle_attention_outgoing(x, edges = x, mask = mask) + x
        x = self.triangle_multiply_ingoing(x, mask = mask) + x
        x = self.triangle_multiply_outgoing(x, mask = mask) + x
        return x

class MsaAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.
    ):
        super().__init__()
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False)
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None
    ):
        x = self.row_attn(x, mask = mask) + x
        x = self.col_attn(x, mask = mask, edges = pairwise_repr) + x
        return x

# main class

class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PairwiseAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout),
                MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout),
            ]))

    def forward(
        self,
        x,
        m,
        mask = None,
        msa_mask = None
    ):
        for attn, ff, msa_attn, msa_ff in self.layers:
            # msa attention and transition

            m = msa_attn(m, mask = msa_mask, pairwise_repr = x) + m
            m = msa_ff(m) + m

            # pairwise attention and transition

            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return x, m