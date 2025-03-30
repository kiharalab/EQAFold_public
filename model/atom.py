import numpy  as np
import torch
import torch.nn.functional as F
from model import r3
import jax
import sys
sys.path.append('./dataset/af2_util')
import residue_constants

def batched_gather(params, indices, axis=0, batch_dims=0):
  """Implements a JAX equivalent of `tf.gather` with `axis` and `batch_dims`."""
  take_fn = lambda p, i: np.take(p, i, axis=axis)
  for _ in range(batch_dims):
    take_fn = jax.vmap(take_fn)
  return take_fn(params, indices)

def torsion_angles_to_frames(
    aatype: np.ndarray,  # (N)
    backb_to_global: r3.Rigids,  # (N)
    torsion_angles_sin_cos: np.ndarray  # (N, 7, 2)
) -> r3.Rigids:  # (N, 8)
  """Compute rigid group frames from torsion angles.

  Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" lines 2-10
  Jumper et al. (2021) Suppl. Alg. 25 "makeRotX"

  Args:
    aatype: aatype for each residue
    backb_to_global: Rigid transformations describing transformation from
      backbone frame to global frame.
    torsion_angles_sin_cos: sin and cosine of the 7 torsion angles
  Returns:
    Frames corresponding to all the Sidechain Rigid Transforms
  """
  assert len(aatype.shape) == 1
  #assert len(backb_to_global.rot.xx.shape) == 1
  assert len(torsion_angles_sin_cos.shape) == 3
  assert torsion_angles_sin_cos.shape[1] == 7
  assert torsion_angles_sin_cos.shape[2] == 2

  # Gather the default frames for all rigid groups.
  # r3.Rigids with shape (N, 8)
  m = batched_gather(residue_constants.restype_rigid_group_default_frame,
                           aatype)
  m = torch.tensor(m)
  default_frames = r3.rigids_from_tensor4x4(m)

  # Create the rotation matrices according to the given angles (each frame is
  # defined such that its rotation is around the x-axis).
  sin_angles = torsion_angles_sin_cos[..., 0]
  cos_angles = torsion_angles_sin_cos[..., 1]

  # insert zero rotation for backbone group.
  num_residues, = aatype.shape
  sin_angles = torch.cat([torch.zeros([num_residues, 1]), sin_angles],
                               dim=-1)
  cos_angles = torch.cat([torch.ones([num_residues, 1]), cos_angles],
                               dim=-1)
  zeros = torch.zeros_like(sin_angles)
  ones = torch.ones_like(sin_angles)

  # all_rots are r3.Rots with shape (N, 8)
  all_rots = r3.Rots(ones, zeros, zeros,
                     zeros, cos_angles, -sin_angles,
                     zeros, sin_angles, cos_angles)

  # Apply rotations to the frames.
  all_frames = r3.rigids_mul_rots(default_frames, all_rots)

  # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
  # the previous frame. So chain them up accordingly.
  chi2_frame_to_frame = jax.tree_map(lambda x: x[:, 5], all_frames)
  chi3_frame_to_frame = jax.tree_map(lambda x: x[:, 6], all_frames)
  chi4_frame_to_frame = jax.tree_map(lambda x: x[:, 7], all_frames)

  chi1_frame_to_backb = jax.tree_map(lambda x: x[:, 4], all_frames)
  chi2_frame_to_backb = r3.rigids_mul_rigids(chi1_frame_to_backb,
                                             chi2_frame_to_frame)
  chi3_frame_to_backb = r3.rigids_mul_rigids(chi2_frame_to_backb,
                                             chi3_frame_to_frame)
  chi4_frame_to_backb = r3.rigids_mul_rigids(chi3_frame_to_backb,
                                             chi4_frame_to_frame)

  # Recombine them to a r3.Rigids with shape (N, 8).
  def _concat_frames(xall, x5, x6, x7):
    return torch.cat(
        [xall[:, 0:5], x5[:, None], x6[:, None], x7[:, None]], dim=-1)

  all_frames_to_backb = jax.tree_map(
      _concat_frames,
      all_frames,
      chi2_frame_to_backb,
      chi3_frame_to_backb,
      chi4_frame_to_backb)

  # Create the global frames.
  # shape (N, 8)
  
  #print(r3.rigids_to_tensor_flat12(backb_to_global))
  #print(len(all_frames_to_backb.rot[0]))
  '''
  all_frames_to_global = r3.rigids_mul_rigids(
      jax.tree_map(lambda x: x[:, None], backb_to_global),
      all_frames_to_backb)
  '''
  all_frames_to_global = r3.rigids_mul_rigids(
      jax.tree_map(lambda x: x.unsqueeze(-1), backb_to_global),
      all_frames_to_backb)
  return all_frames_to_global

def frames_and_literature_positions_to_atom14_pos(
    aatype: np.ndarray,  # (N)
    all_frames_to_global: r3.Rigids  # (N, 8)
) -> r3.Vecs:  # (N, 14)
  """Put atom literature positions (atom14 encoding) in each rigid group.

  Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

  Args:
    aatype: aatype for each residue.
    all_frames_to_global: All per residue coordinate frames.
  Returns:
    Positions of all atom coordinates in global frame.
  """

  # Pick the appropriate transform for every atom.
  residx_to_group_idx = batched_gather(
      residue_constants.restype_atom14_to_rigid_group, aatype)
  group_mask = F.one_hot(
      torch.tensor(residx_to_group_idx), num_classes=8)  # shape (N, 14, 8)

  # r3.Rigids with shape (N, 14)
  '''
  map_atoms_to_global = jax.tree_map(
      lambda x: torch.sum(x[:, None, :] * group_mask, dim=-1),
      all_frames_to_global)
  '''
  map_atoms_to_global = jax.tree_map(
      lambda x: torch.sum(x.unsqueeze(-2) * group_mask, dim=-1),
      all_frames_to_global)
  # Gather the literature atom positions for each residue.
  # r3.Vecs with shape (N, 14)
  lit_positions = r3.vecs_from_tensor(
      batched_gather(
          residue_constants.restype_atom14_rigid_group_positions, aatype))

  # Transform each atom from its local frame to the global frame.
  # r3.Vecs with shape (N, 14)
  pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)

  # Mask out non-existing atoms.
  mask = batched_gather(residue_constants.restype_atom14_mask, aatype)
  pred_positions = jax.tree_map(lambda x: x * torch.tensor(mask), pred_positions)

  return pred_positions