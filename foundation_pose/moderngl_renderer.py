import logging
import math
from dataclasses import dataclass

import moderngl
import numpy as np
import torch

from Utils import glcam_in_cvcam, projection_matrix_from_intrinsics


SOFTWARE_RENDERER_KEYWORDS = ("llvmpipe", "softpipe", "swrast")

INSTANCE_NAMES = ("a_m0", "a_m1", "a_m2", "a_m3", "a_o0", "a_o1", "a_o2", "a_o3", "a_tile")
INSTANCE_SLOTS = len(INSTANCE_NAMES)

VERTEX_SHADER = """
#version 330
in vec3 in_pos;
in vec3 in_normal;
in vec3 in_vc;
in vec4 a_m0;
in vec4 a_m1;
in vec4 a_m2;
in vec4 a_m3;
in vec4 a_o0;
in vec4 a_o1;
in vec4 a_o2;
in vec4 a_o3;
in vec4 a_tile;
uniform float u_cols;
uniform float u_rows;
out vec4 v_clipd;
out vec3 v_xyz;
out vec3 v_vc;
out float v_diffuse;
void main() {
    mat4 M = mat4(a_m0, a_m1, a_m2, a_m3);
    mat4 O = mat4(a_o0, a_o1, a_o2, a_o3);
    vec4 p = vec4(in_pos, 1.0);
    vec4 clip = M * p;
    v_clipd = vec4(clip.w + clip.x, clip.w - clip.x, clip.w + clip.y, clip.w - clip.y);
    clip.x = clip.x / u_cols + (-1.0 + (2.0 * a_tile.x + 1.0) / u_cols) * clip.w;
    clip.y = clip.y / u_rows + (-1.0 + (2.0 * a_tile.y + 1.0) / u_rows) * clip.w;
    gl_Position = clip;
    v_xyz = (O * p).xyz;
    v_vc = in_vc;
    v_diffuse = clamp(dot(normalize(mat3(O) * in_normal), vec3(0.0, 0.0, -1.0)), 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
in vec4 v_clipd;
in vec3 v_xyz;
in vec3 v_vc;
in float v_diffuse;
uniform float u_use_light;
out vec4 o_rgb;
out vec4 o_xyz;
void main() {
    if (any(lessThan(v_clipd, vec4(0.0)))) {
        discard;
    }
    vec3 color = v_vc;
    if (u_use_light > 0.5) {
        color = v_vc * 0.8 + v_diffuse * v_vc * 0.5;
    }
    o_rgb = vec4(clamp(color, 0.0, 1.0), 1.0);
    o_xyz = vec4(v_xyz, 1.0);
}
"""


def is_software_renderer(gl_renderer: str) -> bool:
  name = gl_renderer.lower()
  return any(keyword in name for keyword in SOFTWARE_RENDERER_KEYWORDS)


@dataclass
class _MeshBuffers:
  ref: torch.Tensor
  vao: moderngl.VertexArray
  vbo_color: moderngl.Buffer


class ModernglBatchRenderer:
  def __init__(self, backend: str = "egl", max_tiles_per_pass: int = 1024):
    self.max_tiles_per_pass = max_tiles_per_pass
    self.ctx = moderngl.create_standalone_context(backend=backend)
    self.ctx.enable(moderngl.DEPTH_TEST)
    self.gl_renderer = self.ctx.info["GL_RENDERER"]
    logging.info(f"moderngl GL_RENDERER: {self.gl_renderer}")
    if is_software_renderer(self.gl_renderer):
      logging.warning(f"moderngl is using a software renderer ({self.gl_renderer}); rendering will be slow")
    self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    self.inst_bufs = [self.ctx.buffer(reserve=max_tiles_per_pass * 16) for _ in range(INSTANCE_SLOTS)]
    self._meshes = {}
    self._fbo = None
    self._fbo_size = None

  def render(self, K, H, W, ob_in_cams: torch.Tensor, mesh_tensors: dict, output_size=None,
             bbox2d: torch.Tensor | None = None, use_light: bool = True, extra: dict | None = None
             ) -> tuple[torch.Tensor, torch.Tensor, None]:
    if "tex" in mesh_tensors:
      raise NotImplementedError("ModernglBatchRenderer does not support textured meshes")
    device = ob_in_cams.device
    N = ob_in_cams.shape[0]
    h, w = (H, W) if output_size is None else (int(output_size[0]), int(output_size[1]))

    poses = ob_in_cams.to(dtype=torch.float32)
    proj = torch.as_tensor(projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100),
                           dtype=torch.float32, device=device)
    glcam = torch.as_tensor(glcam_in_cvcam, dtype=torch.float32, device=device)
    mtx = proj @ (glcam @ poses)
    if bbox2d is not None:
      bbox2d = bbox2d.to(device=device, dtype=torch.float32)
      l = bbox2d[:, 0]
      t = H - bbox2d[:, 1]
      r = bbox2d[:, 2]
      b = H - bbox2d[:, 3]
      tf = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).expand(N, 4, 4).clone()
      tf[:, 0, 0] = W / (r - l)
      tf[:, 1, 1] = H / (t - b)
      tf[:, 3, 0] = (W - r - l) / (r - l)
      tf[:, 3, 1] = (H - t - b) / (t - b)
      mtx = tf.transpose(1, 2) @ mtx

    mesh = self._mesh_buffers(mesh_tensors)
    mesh.vbo_color.write(np.ascontiguousarray(mesh_tensors["vertex_color"].detach().cpu().numpy(),
                                              dtype=np.float32).tobytes())

    rgb_out = []
    xyz_out = []
    for start in range(0, N, self.max_tiles_per_pass):
      sl = slice(start, min(start + self.max_tiles_per_pass, N))
      rgb, xyz = self._render_pass(mesh, mtx[sl], poses[sl], h, w, use_light)
      rgb_out.append(rgb)
      xyz_out.append(xyz)
    rgb = torch.cat(rgb_out, dim=0).to(device)
    xyz = torch.cat(xyz_out, dim=0).to(device)
    depth = xyz[..., 2]
    rgb = rgb * (depth > 0)[..., None]
    if extra is not None:
      extra["xyz_map"] = xyz
    return rgb, depth, None

  def _render_pass(self, mesh, mtx, poses, h, w, use_light):
    n = mtx.shape[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    inst = np.zeros((n, INSTANCE_SLOTS, 4), dtype=np.float32)
    inst[:, 0:4] = mtx.transpose(1, 2).reshape(n, 4, 4).cpu().numpy()
    inst[:, 4:8] = poses.transpose(1, 2).reshape(n, 4, 4).cpu().numpy()
    tile_idx = np.arange(n)
    inst[:, 8, 0] = tile_idx % cols
    inst[:, 8, 1] = tile_idx // cols
    for slot, buf in enumerate(self.inst_bufs):
      buf.write(np.ascontiguousarray(inst[:, slot]).tobytes())

    fbo = self._framebuffer(cols * w, rows * h)
    fbo.use()
    fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    self.prog["u_cols"].value = float(cols)
    self.prog["u_rows"].value = float(rows)
    self.prog["u_use_light"].value = 1.0 if use_light else 0.0
    mesh.vao.render(moderngl.TRIANGLES, instances=n)

    rgb_raw = np.frombuffer(fbo.read(attachment=0, components=4, dtype="f4"), dtype=np.float32)
    xyz_raw = np.frombuffer(fbo.read(attachment=1, components=4, dtype="f4"), dtype=np.float32)
    rgb = self._untile(rgb_raw, rows, cols, h, w, n)
    xyz = self._untile(xyz_raw, rows, cols, h, w, n)[..., :3]
    return torch.from_numpy(rgb[..., :3].copy()), torch.from_numpy(xyz.copy())

  def _untile(self, raw, rows, cols, h, w, n):
    grid = raw.reshape(rows, h, cols, w, 4).transpose(0, 2, 1, 3, 4).reshape(rows * cols, h, w, 4)
    return np.ascontiguousarray(grid[:n, ::-1])

  def _framebuffer(self, width, height):
    if self._fbo_size != (width, height):
      if self._fbo is not None:
        self._fbo.release()
      color = [self.ctx.texture((width, height), 4, dtype="f4") for _ in range(2)]
      depth = self.ctx.depth_renderbuffer((width, height))
      self._fbo = self.ctx.framebuffer(color_attachments=color, depth_attachment=depth)
      self._fbo_size = (width, height)
    return self._fbo

  def _mesh_buffers(self, mesh_tensors):
    pos = mesh_tensors["pos"]
    key = (pos.data_ptr(), tuple(pos.shape), mesh_tensors["faces"].data_ptr())
    if key in self._meshes:
      return self._meshes[key]
    pos_np = np.ascontiguousarray(pos.detach().cpu().numpy(), dtype=np.float32)
    normal_np = np.ascontiguousarray(mesh_tensors["vnormals"].detach().cpu().numpy(), dtype=np.float32)
    faces_np = np.ascontiguousarray(mesh_tensors["faces"].detach().cpu().numpy(), dtype=np.int32)
    vbo_pos = self.ctx.buffer(pos_np.tobytes())
    vbo_normal = self.ctx.buffer(normal_np.tobytes())
    vbo_color = self.ctx.buffer(reserve=pos_np.nbytes)
    ibo = self.ctx.buffer(faces_np.tobytes())
    content = [
      (vbo_pos, "3f", "in_pos"),
      (vbo_normal, "3f", "in_normal"),
      (vbo_color, "3f", "in_vc"),
    ]
    content += [(buf, "4f/i", name) for buf, name in zip(self.inst_bufs, INSTANCE_NAMES)]
    vao = self.ctx.vertex_array(self.prog, content, index_buffer=ibo, index_element_size=4)
    entry = _MeshBuffers(ref=pos, vao=vao, vbo_color=vbo_color)
    self._meshes[key] = entry
    return entry
