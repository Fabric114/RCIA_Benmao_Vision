import torch
import torch.nn.functional as F


def _skew(v: torch.Tensor) -> torch.Tensor:
  x, y, z = v.unbind(-1)
  zero = torch.zeros_like(x)
  return torch.stack([
    torch.stack([zero, -z, y], dim=-1),
    torch.stack([z, zero, -x], dim=-1),
    torch.stack([-y, x, zero], dim=-1),
  ], dim=-2)


def so3_exp_map(log_rot: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
  '''Rodrigues formula, same result as pytorch3d.transforms.so3_exp_map. (N,3) -> (N,3,3)'''
  theta_sq = (log_rot ** 2).sum(-1)
  small = theta_sq < eps ** 2
  safe_theta_sq = torch.where(small, torch.ones_like(theta_sq), theta_sq)
  theta = torch.sqrt(safe_theta_sq)
  A = torch.where(small, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta)
  B = torch.where(small, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / safe_theta_sq)
  W = _skew(log_rot)
  eye = torch.eye(3, dtype=log_rot.dtype, device=log_rot.device).expand_as(W)
  return eye + A[..., None, None] * W + B[..., None, None] * (W @ W)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  '''Gram-Schmidt on the two 3-vectors, same result as pytorch3d.transforms.rotation_6d_to_matrix. (...,6) -> (...,3,3)'''
  a1, a2 = d6[..., :3], d6[..., 3:]
  b1 = F.normalize(a1, dim=-1)
  b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
  b3 = torch.cross(b1, b2, dim=-1)
  return torch.stack((b1, b2, b3), dim=-2)
