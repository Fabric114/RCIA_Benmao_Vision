import torch
import torch.nn.functional as F


def _quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
  r, i, j, k = torch.unbind(quaternions, -1)
  two_s = 2.0 / (quaternions * quaternions).sum(-1)
  o = torch.stack((
    1 - two_s * (j * j + k * k),
    two_s * (i * j - k * r),
    two_s * (i * k + j * r),
    two_s * (i * j + k * r),
    1 - two_s * (i * i + k * k),
    two_s * (j * k - i * r),
    two_s * (i * k - j * r),
    two_s * (j * k + i * r),
    1 - two_s * (i * i + j * j),
  ), -1)
  return o.reshape(quaternions.shape[:-1] + (3, 3))


def so3_exp_map(log_rot: torch.Tensor) -> torch.Tensor:
  '''Ported from pytorch3d 0.7.9 (axis_angle -> quaternion -> matrix), bit-exact. (N,3) -> (N,3,3)'''
  angles = torch.norm(log_rot, p=2, dim=-1, keepdim=True)
  sin_half_angles_over_angles = 0.5 * torch.sinc(angles * 0.5 / torch.pi)
  quaternions = torch.cat([torch.cos(angles * 0.5), log_rot * sin_half_angles_over_angles], dim=-1)
  return _quaternion_to_matrix(quaternions)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  '''Gram-Schmidt on the two 3-vectors, same result as pytorch3d.transforms.rotation_6d_to_matrix. (...,6) -> (...,3,3)'''
  a1, a2 = d6[..., :3], d6[..., 3:]
  b1 = F.normalize(a1, dim=-1)
  b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
  b3 = torch.cross(b1, b2, dim=-1)
  return torch.stack((b1, b2, b3), dim=-2)
