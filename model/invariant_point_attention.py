import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum

from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from model import quat_affine
import numpy as np

import rmsd
# helpers
from .resnet import BasicBlock1D
from side_chain_fape import SC_FAPELoss
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
        eps = 1e-8,
        args=None
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        self.require_pairwise_repr = require_pairwise_repr

        # num attention contributions

        num_attn_logits = 3 if require_pairwise_repr else 2

        # qkv projection for scalar attention (normal)

        self.scalar_attn_logits_scale = (num_attn_logits * scalar_key_dim) ** -0.5

        self.to_scalar_q = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_k = nn.Linear(dim, scalar_key_dim * heads, bias = False)
        self.to_scalar_v = nn.Linear(dim, scalar_value_dim * heads, bias = False)

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

        self.num_head = heads
        self.num_point_qk = point_key_dim
        self.num_point_v = point_value_dim
        self.args = args

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
        #print(x.size())
        q_scalar, k_scalar, v_scalar = self.to_scalar_q(x), self.to_scalar_k(x), self.to_scalar_v(x)
        q_point, k_point, v_point = self.to_point_q(x), self.to_point_k(x), self.to_point_v(x)

        # split out heads

        q_scalar, k_scalar, v_scalar = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q_scalar, k_scalar, v_scalar))
        q_point, k_point, v_point = map(lambda t: rearrange(t, 'b n (h d c) -> (b h) n d c', h = h, c = 3), (q_point, k_point, v_point))

        #==========================================================================
        q_point_local =  rearrange(q_point, 'b n d r -> r n (b d) ') 
        k_point_local =  rearrange(k_point, 'b n d r -> r n (b d) ') 
        v_point_local =  rearrange(v_point, 'b n d r -> r n (b d) ') 
        #print('q point local: ', q_point_local.size())
        q_point_global = torch.stack(affine.apply_to_point(q_point_local.cpu(), extra_dims=1)).squeeze(1).cuda(self.args.device_id)
        #print('q point global: ', q_point_global.size())
        k_point_global = torch.stack(affine.apply_to_point(k_point_local.cpu(), extra_dims=1)).squeeze(1).cuda(self.args.device_id)
        v_point_global = torch.stack(affine.apply_to_point(v_point_local.cpu(), extra_dims=1)).squeeze(1).cuda(self.args.device_id)
        
        #q_point_global = torch.stack(affine.apply_to_point(q_point_local.cpu(), extra_dims=1)).cuda(self.args.device_id)
        #k_point_global = torch.stack(affine.apply_to_point(k_point_local.cpu(), extra_dims=1)).cuda(self.args.device_id)
        #v_point_global = torch.stack(affine.apply_to_point(v_point_local.cpu(), extra_dims=1)).cuda(self.args.device_id)
        #print(torch.stack(q_point_global).squeeze(1)[0][0][:20])
        #print('='*80)
        q_point_global = rearrange(q_point_global, 'r n (b d)  -> b n d r ', b=b*h) 
        k_point_global = rearrange(k_point_global, 'r n (b d)  -> b n d r ', b=b*h) 
        v_point_global = rearrange(v_point_global, 'r n (b d)  -> b n d r ', b=b*h) 
     

        #v_point = v_point.to(compute_type) ###[FIXXX]: changed the type here to test
        q_point, k_point, v_point = q_point_global.to(compute_type) , k_point_global.to(compute_type) , v_point_global.to(compute_type) 
        #print(q_point.size())
        #==========================================================================
        # derive attn logits for scalar and pairwise

        attn_logits_scalar = einsum('b i d, b j d -> b i j', q_scalar, k_scalar) * self.scalar_attn_logits_scale
        #print(attn_logits_scalar.size())
        if require_pairwise_repr:
            #print(pairwise_repr.size())
            attn_logits_pairwise = self.to_pairwise_attn_bias(pairwise_repr) * self.pairwise_attn_logits_scale
            #print(attn_logits_pairwise.size())
        # derive attn logits for point attention

        point_qk_diff = rearrange(q_point, 'b i d c -> b i () d c') - rearrange(k_point, 'b j d c -> b () j d c')
        point_dist = (point_qk_diff ** 2).sum(dim = -2)

        point_weights = F.softplus(self.point_weights)
        point_weights = repeat(point_weights, 'h -> (b h) () () ()', b = b)

        #print(point_dist.size())
        #print(point_weights.size())
        #print(b, h)
        attn_logits_points = -0.5 * (point_dist * point_weights * self.point_attn_logits_scale).sum(dim = -1)

        # combine attn logits

        attn_logits = attn_logits_scalar + attn_logits_points

        if require_pairwise_repr:
            #print(attn_logits.size())
            #print(attn_logits_pairwise.size())
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
            #print("Types Scalar: {},{}".format(attn.dtype,v_scalar.dtype))

            results_scalar = einsum('b i j, b j d -> b i d', attn, v_scalar)

            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h = h)

            if require_pairwise_repr:
                #print("Types Scalar: {},{}".format(attn_with_heads.dtype,pairwise_repr.dtype))
                results_pairwise = einsum('b h i j, b i j d -> b h i d', attn_with_heads, pairwise_repr)

            # aggregate point values

            # print("Types points: {},{}".format(attn.dtype,v_point.dtype))

            #print(attn.size())
            #print(v_point.size())
            results_points_global = einsum('b i j, b j d c -> b i d c', attn, v_point)

            # rotate aggregated point values back into local frame
            bv, nv, dv, rv =results_points_global.size()#12 ,60, 8, 3
            #results_points = einsum('b n d c, b n c r -> b n d r', results_points - translations, rotations.transpose(-1, -2))
            #==============================================================================================
            results_points =  rearrange(results_points_global, 'b n d r -> r n (b d) ') 
            results_points = torch.stack(affine.invert_point(results_points.cpu(), extra_dims=1)).squeeze(1).cuda(self.args.device_id)
            #print(results_points.size())
            results_points = rearrange(results_points, 'r n (b d)  -> b n d r ', b=bv) 
            results_points = results_points.to(compute_type) 
            #==============================================================================================
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
def generate_new_affine(n):
  #compute_type = torch.float64
  num_residues = n
  quaternion = np.tile(
      np.reshape(np.asarray([1., 0., 0., 0.]), [1, 4]),
      [num_residues, 1])
  quaternion = torch.tensor(quaternion)#.to(compute_type)
  translation = np.zeros([num_residues, 3])
  translation = torch.tensor(translation)#.to(compute_type)
  #translation = np.ones([num_residues, 3]) * 10
  return quat_affine.QuatAffine(quaternion, translation, unstack_inputs=True)

class IPATransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        num_tokens = None,
        predict_points = False,
        pairwise_repr_dim_default=128,
        args=None,
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
                IPABlock(dim = dim, pairwise_repr_dim_default=pairwise_repr_dim_default, args=args, **kwargs),
                nn.Linear(dim, 6)
            ]))

        # output
        
        self.predict_points = predict_points
        self.point_scale = args.point_scale
        if predict_points:
            self.to_points = nn.Linear(dim, 3)
        
        self.angle_channels = 128
        self.transit = nn.Sequential(
				nn.Linear(384, self.angle_channels),
				nn.ReLU()
		)
        self.transit_initial = nn.Sequential(
				nn.Linear(384, self.angle_channels),
				nn.ReLU()
		)
        self.angle_resnet = nn.ModuleList([BasicBlock1D(angle_channel=self.angle_channels), BasicBlock1D(angle_channel=self.angle_channels)])
        self.angle_predict = nn.Sequential(
				nn.Linear(self.angle_channels, 14),
				nn.ReLU()
		)
        self.sc_fape = SC_FAPELoss()
        
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
        aatype=None,
        gt_frames=None,
        atom14=None
    ):
        # print(single_repr)
        compute_type = torch.float32
        x, device, quaternion_multiply, quaternion_to_matrix = single_repr, single_repr.device, self.quaternion_multiply, self.quaternion_to_matrix
        b, n, *_ = x.shape

        affine = generate_new_affine(n)
        affine_true = generate_new_affine(n)
        #print(n)

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
        def attach(x):
            x.requires_grad = True
            return x
        #rotations = quaternion_to_matrix(quaternions)
        #with torch.no_grad():
        
        fape_loss, dis_loss=[],[]
        s_inital = x
        for itr, (block, to_update) in enumerate(self.layers):   
            x = block(
                x,
                pairwise_repr = pairwise_repr,
                rotations = None,
                translations = None,
                mask=masks,
                affine=affine
            )
            
            # update quaternion and translation
            #=========================================
            #affine = quat_affine.QuatAffine.from_tensor(affine.to_tensor())
            act = to_update(x)
            #act_np = act.detach().cpu().numpy()
            rot = affine.rotation
            rot = [torch.stack(a) for a in rot]
            rot = torch.stack(rot)
            #print('rotatation: ', rot.size())
            trans = affine.translation
            trans = torch.stack(trans)
            #print('rotation: ', rot.size())
            #print('trans: ', trans.size())
            affine = affine.pre_compose(act)

            rot = affine.rotation
            rot = [torch.stack(a) for a in rot]
            rot = torch.stack(rot)
            #print('rotatation: ', rot.size())
            trans = affine.translation
            trans = torch.stack(trans)
            #print('rotation: ', rot.size())
            #print('trans: ', trans.size())
            #print('translation: ', trans.size())
            quaternions = affine.quaternion.to(device=device)
            
            #quaternions[:, :, 0] = 1
            #affine.quaternion = quaternions.cpu()
            #print(quaternions[:, 0, :])
            rotations = rot.to(device=device).permute(2,3,0,1).to(torch.float32)
            translations = trans.to(device=device).permute(1,2,0).to(torch.float32)
            
            points_local = translations
            
            #side chains
            act = self.transit(x) + self.transit_initial(s_inital) #963
            act = self.angle_resnet[0](act)
            act = self.angle_resnet[1](act)
            angles = torch.reshape(self.angle_predict(act), (b, n, 7, 2))

            normalized_angles = F.normalize(angles, p=2.0, dim=-1)
            
            if not itr == len(self.layers) - 1:
                affine = affine.apply_rotation_tensor_fn(detach)
            if use_fape:
                T_pred = (rotations, translations)
                #rotations,translations = backbone_frames
         
                #affine_true = quat_affine.QuatAffine.from_tensor(affine_true.to_tensor())
                
                affine_pred = quat_affine.QuatAffine(quaternions[masks].unsqueeze(0).cpu(), translations[masks].unsqueeze(0).cpu().permute(2,0,1))
                                            #rotation=rotations[masks].unsqueeze(0).cpu().permute(2,3,0,1))
                #print(affine_pred.rotation)
                R_mask,t_mask = backbone_frames
                T_true_mask = R_mask[masks].unsqueeze(0), t_mask[masks].unsqueeze(0)    #unsqueeze only when batch size is 1
                #affine_true = quat_affine.QuatAffine.from_tensor(affine_true.to_tensor())
                
                quaternion = quat_affine.rot_to_quat(R_mask[masks].unsqueeze(0).cpu().permute(2,3,0,1))
                #affine_true = quat_affine.QuatAffine(torch.tensor(quaternion), t_mask[masks].unsqueeze(0).cpu().permute(2,0,1))
                affine_true = quat_affine.QuatAffine(quaternion, t_mask[masks].unsqueeze(0).cpu().permute(2,0,1))
                R_mask_pred,t_mask_pred = T_pred
                T_pred_mask = R_mask_pred[masks].unsqueeze(0), t_mask_pred[masks].unsqueeze(0)
                '''
                print('='*80)
                print(rotations[0, 0, ...])
                print(R_mask[masks].unsqueeze(0)[0, 0, ...])
                print('='*80)
                '''
                #print(coords[masks].unsqueeze(0)[:,:,1,:])
                #print(t_mask[masks].unsqueeze(0))
                loss = criterion(points_local[masks].unsqueeze(0), coords[masks].unsqueeze(0)[:,:,1,:], affine=affine_pred, affine_true=affine_true, point_scale=self.point_scale)
                fape_loss.append(loss)
               
        if not self.predict_points:
            return x, translations, quaternions

        #print(translations)
        if use_fape:
            # print(fape_loss)
            fape_loss = torch.stack((fape_loss))
            fape_loss = fape_loss.to(device=device)
            
            sc_loss = self.sc_fape(affine=affine_pred, affine_true=affine_true, point_scale=self.point_scale, normalized_angles=normalized_angles[masks],
                                        aatype=aatype, gt_frames=gt_frames, atom14=atom14)
            #print(sc_loss)
            #print(f'fape: {torch.mean(fape_loss)}')
        
            #x_ij = (translations*(self.point_scale))[masks].unsqueeze(0).sub(coords[masks].unsqueeze(0)[:,:,1,:]).norm(p=2, dim=-1)
            #print(f"mean mse loss : {x_ij.mean()}")
            return translations*(self.point_scale), torch.mean(fape_loss), sc_loss
        else:
            return translations, _
