import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from model import quat_affine
import numpy as np

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value

def generate_new_affine(n):
  num_residues = n
  quaternion = np.tile(
      np.reshape(np.asarray([1., 0., 0., 0.]), [1, 4]),
      [num_residues, 1])

  translation = np.zeros([num_residues, 3])
  return quat_affine.QuatAffine(torch.tensor(quaternion), torch.tensor(translation), unstack_inputs=True)
# classes

class InvariantPointAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 12,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 8,
        #pairwise_repr_dim_default = 64,
        pairwise_repr_dim_default = 128,
        pairwise_repr_dim = None,
        require_pairwise_repr = True,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr

        self.num_head = heads
        self.num_point_qk = point_key_dim
        self.num_point_v = point_value_dim

        # num attention contributions

        num_attn_logits = 3 if require_pairwise_repr else 2

        # qkv projection for scalar attention (normal)

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.q_scalar = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.kv_scalar = nn.Linear(dim, self.num_head * (self.num_scalar_v + self.num_scalar_qk), bias = False)

        self.kv_point_local = nn.Linear(dim, self.num_head * 3 * (self.num_point_qk + self.num_point_v), bias = False)
        #self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias = False)

        # qkv projection for point attention (coordinate and orientation aware)

        point_weight_init_value = torch.log(torch.exp(torch.full((heads,), 1.)) - 1.)
        self.point_weights = nn.Parameter(point_weight_init_value)

        self.point_attn_logits_scale = ((num_attn_logits * point_key_dim) * (9 / 2)) ** -0.5

        self.to_point_q = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_k = nn.Linear(dim, point_key_dim * heads * 3, bias = False)
        self.to_point_v = nn.Linear(dim, point_value_dim * heads * 3, bias = False)

        # pairwise representation projection to attention bias

        pairwise_repr_dim = default(pairwise_repr_dim, pairwise_repr_dim_default) if require_pairwise_repr else 0

        if require_pairwise_repr:
            self.pairwise_attn_logits_scale = num_attn_logits ** -0.5

            self.to_pairwise_attn_bias = nn.Sequential(
                nn.Linear(pairwise_repr_dim, heads),
                Rearrange('b ... h -> (b h) ...')
            )

        # combine out - scalar dim + pairwise dim + point dim * (3 for coordinates in R3 and then 1 for norm)

        self.to_out = nn.Linear(heads * (scalar_value_dim + pairwise_repr_dim + point_value_dim * (3 + 1)), dim)

        
        
    def forward(
        self,
        single_repr,
        pairwise_repr = None,
        *,
        rotations,
        translations,
        mask = None,
        affine=None
    ):
        compute_type = torch.float32
        x, b, h, eps, require_pairwise_repr = single_repr, single_repr.shape[0], self.heads, self.eps, self.require_pairwise_repr
        
        assert not (require_pairwise_repr and not exists(pairwise_repr)), 'pairwise representation must be given as second argument'

        # get queries, keys, values for scalar and point (coordinate-aware) attention pathways

        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(x), self.to_scalar_v(x)

        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(x)

        #======================================
        '''
        num_residues = x.shape[1]
        q_point_local = q_point
        q_point_local = np.split(q_point_local, 3, axis=-1)
        # Project query points into global frame.
        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        # Reshape query point for later use.
        q_point = [
            np.reshape(x, [num_residues, self.num_head, self.num_point_qk])
            for x in q_point_global]
        
        kv_point_local = np.split(kv_point_local, 3, axis=-1)
        # Project key and value points into global frame.
        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [
            jnp.reshape(x, [num_residues,
                            num_head, (num_point_qk + num_point_v)])
            for x in kv_point_global]
        # Split key and value points.
        k_point, v_point = list(
            zip(*[
                jnp.split(x, [num_point_qk,], axis=-1)
                for x in kv_point_global
            ]))
        '''
        #======================================
        # split out heads

        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h = h, c = 3), (q_point, k_point, v_point))

        rotations = repeat(rotations, 'b n r1 r2 -> (b h) n r1 r2', h = h).to(compute_type) ###[FIXXX]: changed the type here to test
        translations = repeat(translations, 'b n c -> (b h) n () c', h = h).to(compute_type) ###[FIXXX]: changed the type here to test

        # rotate qkv points into global frame
        #q_point = q_point.to(torch.float16)
        q_point = einsum('b n d c, b n c r -> b n d r', q_point, rotations) + translations
        k_point = einsum('b n d c, b n c r -> b n d r', k_point, rotations) + translations
        v_point = einsum('b n d c, b n c r -> b n d r', v_point, rotations) + translations
        v_point = v_point.to(compute_type) ###[FIXXX]: changed the type here to test

        # derive attn logits for scalar and pairwise

        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale

        if require_pairwise_repr:
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale

        # derive attn logits for point attention

        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2).sum(dim = -2)

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () () ()', b = b)

        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale).sum(dim = -1)

        # combine attn logits

        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            attn_logits = attn_logits + attn_logits_pairwise

        # mask

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)

        # attention

        attn = attn_logits.softmax(dim = - 1).to(compute_type)     ###[FIXXX]: changed the type here to test

        with disable_tf32(), autocast(enabled = False):
            # disable TF32 for precision

            # aggregate values
            # print("Types Scalar: {},{}".format(attn.dtype,v_scalar.dtype))

            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)

            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h = h)

            if require_pairwise_repr:
                results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)

            # aggregate point values

            # print("Types points: {},{}".format(attn.dtype,v_point.dtype))

            results_points = einsum('b i j, b j d c -> b i d c', attn, v_point)

            # rotate aggregated point values back into local frame

            results_points = einsum('b n d c, b n c r -> b n d r', results_points - translations, rotations.transpose(-1, -2))
            results_points_norm = torch.sqrt( torch.square(results_points).sum(dim=-1) + eps )

        # merge back heads

        results_scalar = rearrange(results_scalar, '(b h) n d -> b n (h d)', h = h)
        results_points = rearrange(results_points, '(b h) n d c -> b n (h d c)', h = h)
        results_points_norm = rearrange(results_points_norm, '(b h) n d -> b n (h d)', h = h)

        results = (results_scalar, results_points, results_points_norm)

        if require_pairwise_repr:
            results_pairwise = rearrange(results_pairwise, 'b h n d -> b n (h d)', h = h)
            results = (*results, results_pairwise)

        # concat results and project out

        results = torch.cat(results, dim = -1)
        return self.to_out(results)

# one transformer block based on IPA

def FeedForward(dim, mult = 1., num_layers = 2, act = nn.ReLU):
    layers = []
    dim_hidden = dim * mult

    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden

        layers.append(nn.Linear(dim_in, dim_out))

        if is_last:
            continue

        layers.append(act())

    return nn.Sequential(*layers)

class IPABlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult = 1,
        ff_num_layers = 3,     # in the paper, they used 3 layer transition (feedforward) block
        post_norm = True,      # in the paper, they used post-layernorm - offering pre-norm as well
        pairwise_repr_dim_default=128,
        **kwargs
    ):
        super().__init__()
        self.post_norm = post_norm

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = InvariantPointAttention(dim = dim, pairwise_repr_dim_default=pairwise_repr_dim_default, **kwargs)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult = ff_mult, num_layers = ff_num_layers)

    def forward(self, x, **kwargs):
        post_norm = self.post_norm

        attn_input = x if post_norm else self.attn_norm(x)
        x = self.attn(attn_input, **kwargs) + x
        x = self.attn_norm(x) if post_norm else x

        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.ff_norm(x) if post_norm else x
        return x

# add an IPA Transformer - iteratively updating rotations and translations

# AF2 applies a FAPE auxiliary loss on each layer, as well as a stop gradient on the rotations

class IPATransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens = None,
        predict_points = False,
        pairwise_repr_dim_default=128,
        **kwargs
    ):
        super().__init__()

        # using quaternion functions from pytorch3d

        try:
            from pytorch3d.transforms import quaternion_multiply, quaternion_to_matrix, matrix_to_quaternion
            self.quaternion_to_matrix = quaternion_to_matrix
            self.quaternion_multiply = quaternion_multiply
            self.matrix_to_quaternion = matrix_to_quaternion
        except ImportError as err:
            print('unable to import pytorch3d - please install with `conda install pytorch3d -c pytorch3d`')
            raise err

        # embedding
        print("Embedding: {},{}".format(num_tokens, dim))

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None

        # layers

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                IPABlock(dim = dim, pairwise_repr_dim_default=pairwise_repr_dim_default, **kwargs),
                nn.Linear(dim, 6)
            ]))

        # output

        self.predict_points = predict_points

        if predict_points:
            self.to_points = nn.Linear(dim, 3)

    def forward(
        self,
        single_repr,
        pairwise_repr,
        coords,
        backbone_frames,
        criterion,        
        *,
        translations = None,
        quaternions = None,
        masks = None,
        use_fape = False,
        refine_only = False,
    ):
        # print(single_repr)
        compute_type = torch.float32
        x, device, quaternion_multiply, quaternion_to_matrix = single_repr, single_repr.device, self.quaternion_multiply, self.quaternion_to_matrix
        b, n, *_ = x.shape

        affine = generate_new_affine(n)
        
        # if exists(self.token_emb):
        #     x = self.token_emb(x)

        # if no initial quaternions passed in, start from identity
        
        if not exists(quaternions):
            quaternions = torch.tensor([1., 0., 0., 0.], device = device) # initial rotations
            quaternions = repeat(quaternions, 'd -> b n d', b = b, n = n) 
        # if not translations passed in, start from identity

        if not exists(translations):
            translations = torch.zeros((b, n, 3), device = device)
        
        # go through the layers and apply invariant point attention and feedforward
        # print(f"No. of layers : {len(self.layers)}")
       

        def quaternion_multiply_custom(quater,vec):
            aw, ax, ay, az = torch.unbind(quater, -1)
            bw, bx, by, bz = torch.unbind(vec, -1)
            ow=aw
            ox = aw * bx + ax * bw + ay * bz - az * by + ax
            oy = aw * by - ax * bz + ay * bw + az * bx + ay
            oz = aw * bz + ax * by - ay * bx + az * bw + az
            return torch.stack((ow, ox, oy, oz), -1)
        
        def detach(x):
            return x.detach()

        rotations = quaternion_to_matrix(quaternions)
        #with torch.no_grad():
        
        fape_loss, dis_loss=[],[]
        for itr, (block, to_update) in enumerate(self.layers):   
           

            x = block(
                x,
                pairwise_repr = pairwise_repr,
                rotations = rotations,
                translations = translations,
                mask=masks,
                affine=affine
            )
            # update quaternion and translation
            #=========================================
            affine = quat_affine.QuatAffine.from_tensor(affine.to_tensor())
            act = to_update(x)
            #act_np = act.detach().cpu().numpy()
            affine = affine.pre_compose(act)

            rot = affine.rotation
            rot = [torch.stack(a) for a in rot]
            rot = torch.stack(rot)
            trans = affine.translation
            trans = torch.stack(trans)
            quaternions = affine.quaternion.to(device=device)
            rotations = rot.to(device=device).permute(2,3,0,1).to(torch.float32)
            translations = trans.to(device=device).permute(1,2,0).to(torch.float32)
            quaternion_update, translation_update = act.chunk(2, dim = -1)
            #print('='*80)
            #print(np.asarray(affine.quaternion)[0, :5, :])
            

            quaternions = F.pad(quaternions, (1, 0), value = 1.)
            
            #print(quaternions[0, :5, :])
            #=========================================
            #affine = affine.apply_rotation_tensor_fn(detach)
            #print(f"q shape : {quaternion_update.shape}")
            '''
            quaternion_update = F.pad(quaternion_update, (1, 0), value = 1.)
            
            quaternions = quaternion_multiply_custom(quaternions, quaternion_update)
            #print(quaternions[0, :5, :])
            print('='*80)
            # print(f"q after mul: {quaternions}")
            rotations = quaternion_to_matrix(quaternions).to(compute_type)
            translations = translations + einsum('b n c, b n c r -> b n r', translation_update, rotations)
            # print(f"activations shape: {x.shape}, quaternions shape: {quaternions.shape}, translations shape: {translations.shape}")
            '''
            #print('rotataion', rotations.size())
            #print('translation', translations.size())
            points_local = self.to_points(x)
            points_global = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations
            
            #quaternions = quaternions if itr == len(self.layers) - 1 else quaternions.detach()
            #rotations = rotations if itr == len(self.layers) - 1 else rotations.detach()
            #translations = translations if itr == len(self.layers) - 1 else translations.detach()  
            
            if use_fape:
                T_pred = (rotations, translations)
                R_mask,t_mask = backbone_frames
                T_true_mask = R_mask[masks].unsqueeze(0), t_mask[masks].unsqueeze(0)    #unsqueeze only when batch size is 1
                R_mask_pred,t_mask_pred = T_pred
                T_pred_mask = R_mask_pred[masks].unsqueeze(0), t_mask_pred[masks].unsqueeze(0)
                '''
                print('='*80)
                print(rotations[0, 0, ...])
                print(R_mask[masks].unsqueeze(0)[0, 0, ...])
                print('='*80)
                '''
                loss = criterion(T_pred_mask, points_global[masks].unsqueeze(0), T_true_mask, coords[masks].unsqueeze(0)[:,:,1,:])
                fape_loss.append(loss[0])
                dis_loss.append(loss[1])
        if not self.predict_points:
            return x, translations, quaternions

        if use_fape:
            # print(fape_loss)
            fape_loss = torch.stack((fape_loss))
            dis_loss = torch.stack((dis_loss))
            fape_loss = fape_loss.to(device=device)
            print(f"mean mse loss : {torch.mean(dis_loss)}")
            print(f'fape: {torch.mean(fape_loss)}')
            points_local = self.to_points(x)
            #rotations = quaternion_to_matrix(quaternions)
            rotations_gt,translations_gt = backbone_frames
            # print(f"gt quaternions : {self.matrix_to_quaternion(rotations_gt)}")

            #print(points_local.dtype)
            #print(rotations.dtype)
            #print(translations.dtype)
            points_global = einsum('b n c, b n c d -> b n d', points_local, rotations) + translations    
            return points_global, torch.mean(fape_loss)
        else:
            return points_global, _
