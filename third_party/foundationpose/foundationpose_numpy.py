# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""NumPy implementation of the foundationpose_cpp extension.

Provides the same functions as the CUDA/C++ extension built from cpp/, so that
model.py can run on machines without an NVIDIA GPU (e.g. arm64 or macOS
hosts). The helpers are line-by-line ports of foundationpose_binding.cpp and
foundationpose_sampling_helpers.cpp. `RasterizeCpuContext` replaces
`RasterizeCudaContext` and follows the conventions of nvdiffrast's CudaRaster
that the CUDA implementation uses:

- Pixel (x, y) samples clip space at ((x + 0.5) * 2 / W - 1,
  (y + 0.5) * 2 / H - 1), i.e. row 0 is the bottom of the image (y = -1).
- Triangles are not culled; the closest fragment by z/w wins ("less" test).
- Fragments with z/w outside [-1, 1] are clipped.
- The rasterizer output per pixel is (u, v, z/w, triangle_id + 1), where
  (u, v) are the (perspective-correct) barycentrics of the triangle's first
  two vertices, and 0 everywhere for background pixels.
"""

import numpy as np

# Maximum number of candidate fragments processed at once by the CPU
# rasterizer. Bounds peak memory to a few hundred MB.
_MAX_FRAGMENTS_PER_CHUNK = 8_000_000


def _normalized(v):
  return v / np.linalg.norm(v)


def _generate_icosphere(recursion_level):
  """Returns the unit icosphere vertices in the order of the C++ helper."""
  t = (1.0 + np.sqrt(5.0)) / 2.0
  vertices = [
      _normalized(np.array(p, dtype=np.float32))
      for p in [
          (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
          (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
          (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
      ]
  ]
  faces = [
      (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
      (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
      (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
      (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
  ]
  cache = {}

  def middle_point(p1, p2):
    key = (min(p1, p2), max(p1, p2))
    if key not in cache:
      vertices.append(_normalized((vertices[p1] + vertices[p2]) / 2.0))
      cache[key] = len(vertices) - 1
    return cache[key]

  for _ in range(recursion_level):
    new_faces = []
    for v1, v2, v3 in faces:
      a = middle_point(v1, v2)
      b = middle_point(v2, v3)
      c = middle_point(v3, v1)
      new_faces += [(v1, a, c), (v2, b, a), (v3, c, b), (a, b, c)]
    faces = new_faces
  return np.stack(vertices).astype(np.float32)


def _guess_translation(depth, mask, K, min_depth):
  """Returns the object center guessed from the mask, or None."""
  vs, us = np.nonzero(mask > 0)
  if us.size == 0:
    return None
  uc = (us.min() + us.max()) / 2.0
  vc = (vs.min() + vs.max()) / 2.0
  valid = (mask > 0) & (depth >= min_depth)
  if not np.any(valid):
    return None
  zc = np.median(depth[valid])
  return (np.linalg.inv(K) @ np.array([uc, vc, 1.0])) * zc


def make_rotation_grid(n_views, inplane_step_deg):
  """Returns (N, 4, 4) object-in-camera rotations like MakeRotationGrid."""
  cam_in_obs = _generate_icosphere(2)[:n_views]
  inplane_step = inplane_step_deg / 180.0 * np.pi
  # Same accumulation as the C++ `for (double r = 0; r < 2 * M_PI; r += step)`.
  inplane_rots = []
  rot = 0.0
  while rot < 2.0 * np.pi:
    inplane_rots.append(rot)
    rot += inplane_step

  poses = []
  up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
  for position in cam_in_obs:
    z_axis = -_normalized(position)
    x_axis = np.cross(up, z_axis)
    if not np.any(x_axis):
      x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x_axis = _normalized(x_axis)
    y_axis = _normalized(np.cross(z_axis, x_axis))
    cam_in_ob = np.eye(4, dtype=np.float32)
    cam_in_ob[:3, 0] = x_axis
    cam_in_ob[:3, 1] = y_axis
    cam_in_ob[:3, 2] = z_axis
    cam_in_ob[:3, 3] = position
    for inplane_rot in inplane_rots:
      c, s = np.cos(inplane_rot), np.sin(inplane_rot)
      r_inplane = np.eye(4, dtype=np.float32)
      r_inplane[:2, :2] = [[c, -s], [s, c]]
      poses.append(np.linalg.inv(cam_in_ob @ r_inplane))
  return np.stack(poses).astype(np.float32)


def sample_initial_poses(mask, depth, K, n_views, inplane_step_deg):
  """Port of foundationpose_cpp.sample_initial_poses."""
  mask = np.asarray(mask)
  depth = np.asarray(depth, dtype=np.float32)
  K = np.asarray(K, dtype=np.float32)
  center = _guess_translation(depth, mask, K, min_depth=0.1)
  if center is None:
    center = np.array([0.0, 0.0, 0.8], dtype=np.float32)
  poses = make_rotation_grid(n_views, inplane_step_deg)
  poses[:, :3, 3] = center
  return poses


def compute_crop_window_tf(poses, K, out_h, out_w, crop_ratio, mesh_diameter):
  """Port of foundationpose_cpp.compute_crop_window_tf."""
  poses = np.asarray(poses, dtype=np.float32)
  K = np.asarray(K, dtype=np.float32)
  r = mesh_diameter * crop_ratio / 2.0
  offsets = np.array(
      [[0, 0, 0], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]],
      dtype=np.float32,
  )
  pts = poses[:, None, :3, 3] + offsets[None]  # (B, 5, 3)
  projected = pts @ K.T
  uvs = projected[..., :2] / projected[..., 2:3]
  center = uvs[:, 0]
  radius = np.abs(uvs[..., 1] - center[:, None, 1]).max(axis=1)
  # std::round rounds half away from zero, unlike np.round.
  round_half_away = lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5)
  left = round_half_away(center[:, 0] - radius)
  right = round_half_away(center[:, 0] + radius)
  top = round_half_away(center[:, 1] - radius)
  bottom = round_half_away(center[:, 1] + radius)

  tfs = np.zeros((len(poses), 3, 3), dtype=np.float32)
  tfs[:, 0, 0] = out_w / (right - left)
  tfs[:, 0, 2] = -left * out_w / (right - left)
  tfs[:, 1, 1] = out_h / (bottom - top)
  tfs[:, 1, 2] = -top * out_h / (bottom - top)
  tfs[:, 2, 2] = 1.0
  return tfs


def _axis_angle_to_matrix(axis, angle):
  x, y, z = axis
  c, s = np.cos(angle), np.sin(angle)
  C = 1.0 - c
  return np.array(
      [
          [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
          [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
          [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
      ],
      dtype=np.float32,
  )


def update_refined_poses(
    poses, trans_delta, rot_delta, mesh_diameter, rot_normalizer
):
  """Port of foundationpose_cpp.update_refined_poses."""
  result = np.array(poses, dtype=np.float32, copy=True)
  trans_delta = np.asarray(trans_delta, dtype=np.float32)
  rot_delta = np.asarray(rot_delta, dtype=np.float32)
  result[:, :3, 3] += trans_delta * (mesh_diameter / 2.0)
  rot_vecs = np.tanh(rot_delta) * rot_normalizer
  for i, rot_vec in enumerate(rot_vecs):
    norm = np.linalg.norm(rot_vec)
    if norm > 1e-6:
      rot_mat_delta = _axis_angle_to_matrix(rot_vec / norm, norm).T
      result[i, :3, :3] = rot_mat_delta @ result[i, :3, :3]
  return result


def apply_mesh_center_offset(pose, mesh_center):
  """Port of foundationpose_cpp.apply_mesh_center_offset."""
  tf_to_center = np.eye(4, dtype=np.float32)
  tf_to_center[:3, 3] = mesh_center
  return (np.asarray(pose, dtype=np.float32) @ tf_to_center).astype(np.float32)


class RasterizeCpuContext:
  """CPU replacement for foundationpose_cpp.RasterizeCudaContext."""

  def __init__(self, width=160, height=160, max_images=64):
    del width, height, max_images  # Only needed for CUDA buffer allocation.

  def rasterize(self, v_clip, faces, H, W):
    """Rasterizes (N, V, 4) clip-space vertices into (N, H, W, 4)."""
    v_clip = np.asarray(v_clip, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    N, V = v_clip.shape[:2]
    if N <= 0 or V <= 0 or len(faces) <= 0 or H <= 0 or W <= 0:
      raise ValueError(
          f"Invalid dimensions: N={N}, V={V}, F={len(faces)}, H={H}, W={W}"
      )
    if faces.min() < 0 or faces.max() >= V:
      raise ValueError(
          f"Mesh face index out of bounds: min_idx={faces.min()},"
          f" max_idx={faces.max()}, but vertex count V={V}"
      )

    out = np.zeros((N, H, W, 4), dtype=np.float32)
    best_z = np.full((N, H * W), np.inf, dtype=np.float32)
    w = v_clip[..., 3:4]
    ndc = v_clip[..., :3] / w
    # Pixel coordinates in which pixel centers are at integer + 0.5.
    px = (ndc[..., 0] + 1.0) * (W / 2.0)
    py = (ndc[..., 1] + 1.0) * (H / 2.0)
    tri_px = px[:, faces]  # (N, F, 3)
    tri_py = py[:, faces]
    tri_z = ndc[:, faces, 2]
    tri_w = w[:, faces, 0]

    # Signed doubled area; degenerate triangles cover no pixel centers.
    area = (tri_px[..., 1] - tri_px[..., 0]) * (tri_py[..., 2] - tri_py[..., 0]) - (
        tri_px[..., 2] - tri_px[..., 0]
    ) * (tri_py[..., 1] - tri_py[..., 0])
    x0 = np.clip(np.ceil(tri_px.min(-1) - 0.5), 0, W).astype(np.int64)
    x1 = np.clip(np.floor(tri_px.max(-1) - 0.5) + 1, 0, W).astype(np.int64)
    y0 = np.clip(np.ceil(tri_py.min(-1) - 0.5), 0, H).astype(np.int64)
    y1 = np.clip(np.floor(tri_py.max(-1) - 0.5) + 1, 0, H).astype(np.int64)
    bw = np.maximum(x1 - x0, 0)
    bh = np.maximum(y1 - y0, 0)
    counts = np.where(np.abs(area) > 1e-12, bw * bh, 0).reshape(-1)

    nz = np.nonzero(counts)[0]
    if nz.size == 0:
      return out
    cum = np.cumsum(counts[nz])
    start = 0
    while start < nz.size:
      base = cum[start - 1] if start > 0 else 0
      end = int(np.searchsorted(cum, base + _MAX_FRAGMENTS_PER_CHUNK, "right"))
      end = max(end, start + 1)
      self._rasterize_chunk(
          nz[start:end], counts[nz[start:end]], faces.shape[0], W,
          tri_px, tri_py, tri_z, tri_w, area, x0, y0, bw, best_z, out,
      )
      start = end
    return out

  @staticmethod
  def _rasterize_chunk(
      pair_ids, counts, F, W, tri_px, tri_py, tri_z, tri_w, area, x0, y0, bw,
      best_z, out,
  ):
    """Rasterizes the given (image, triangle) pairs into `out`."""
    frag_pair = np.repeat(pair_ids, counts)
    offsets = np.arange(frag_pair.size) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    n_idx, f_idx = np.divmod(frag_pair, F)
    fx = x0[n_idx, f_idx] + offsets % bw[n_idx, f_idx]
    fy = y0[n_idx, f_idx] + offsets // bw[n_idx, f_idx]
    sx = fx + 0.5
    sy = fy + 0.5

    ax, bx, cx = (tri_px[n_idx, f_idx, k] for k in range(3))
    ay, by, cy = (tri_py[n_idx, f_idx, k] for k in range(3))
    a = area[n_idx, f_idx]
    # Screen-space barycentrics of vertices 0, 1, 2.
    b0 = ((bx - sx) * (cy - sy) - (cx - sx) * (by - sy)) / a
    b1 = ((cx - sx) * (ay - sy) - (ax - sx) * (cy - sy)) / a
    b2 = 1.0 - b0 - b1
    inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0)
    z = (
        b0 * tri_z[n_idx, f_idx, 0]
        + b1 * tri_z[n_idx, f_idx, 1]
        + b2 * tri_z[n_idx, f_idx, 2]
    )
    inside &= (z >= -1.0) & (z <= 1.0)
    if not np.any(inside):
      return
    n_idx, f_idx, fx, fy = n_idx[inside], f_idx[inside], fx[inside], fy[inside]
    b0, b1, b2, z = b0[inside], b1[inside], b2[inside], z[inside]

    # Keep the closest fragment per pixel, ties resolved by triangle index.
    pixel = n_idx * (best_z.shape[1]) + fy * W + fx
    order = np.lexsort((f_idx, z, pixel))
    first = np.ones(order.size, dtype=bool)
    first[1:] = pixel[order][1:] != pixel[order][:-1]
    winners = order[first]
    pixel, z = pixel[winners], z[winners]
    closer = z < best_z.reshape(-1)[pixel]
    winners, pixel, z = winners[closer], pixel[closer], z[closer]
    best_z.reshape(-1)[pixel] = z

    # Perspective-correct barycentrics, as returned by nvdiffrast.
    n_w, f_w = n_idx[winners], f_idx[winners]
    p0 = b0[winners] / tri_w[n_w, f_w, 0]
    p1 = b1[winners] / tri_w[n_w, f_w, 1]
    p2 = b2[winners] / tri_w[n_w, f_w, 2]
    total = p0 + p1 + p2
    flat = out.reshape(-1, 4)
    flat[pixel, 0] = p0 / total
    flat[pixel, 1] = p1 / total
    flat[pixel, 2] = z
    flat[pixel, 3] = f_w + 1

  def interpolate(self, attributes, rast, faces, H, W):
    """Interpolates (N, V, C) vertex attributes over the rasterized images."""
    attributes = np.asarray(attributes, dtype=np.float32)
    rast = np.asarray(rast, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    N = attributes.shape[0]
    out = np.zeros((N, H, W, attributes.shape[-1]), dtype=np.float32)
    covered = rast[..., 3] > 0
    n_idx, ys, xs = np.nonzero(covered)
    tri = faces[rast[n_idx, ys, xs, 3].astype(np.int64) - 1]  # (P, 3)
    u = rast[n_idx, ys, xs, 0][:, None]
    v = rast[n_idx, ys, xs, 1][:, None]
    out[n_idx, ys, xs] = (
        u * attributes[n_idx, tri[:, 0]]
        + v * attributes[n_idx, tri[:, 1]]
        + (1.0 - u - v) * attributes[n_idx, tri[:, 2]]
    )
    return out
