# Copyright 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests the NumPy implementation of the foundationpose_cpp extension.

The golden values in testdata/foundationpose_cpp_golden.npz were produced by
the C++ helpers of foundationpose_binding.cpp (compiled without CUDA) for the
inputs stored in the same file.
"""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from third_party.foundationpose import foundationpose_numpy as fp

_GOLDEN = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "testdata",
    "foundationpose_cpp_golden.npz",
)


class CppEquivalenceTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.golden = dict(np.load(_GOLDEN))

  def test_sample_initial_poses(self):
    g = self.golden
    poses = fp.sample_initial_poses(g["mask"], g["depth"], g["K"], 40, 60.0)
    self.assertEqual(poses.dtype, np.float32)
    np.testing.assert_allclose(poses, g["poses"], atol=1e-6)

  def test_compute_crop_window_tf(self):
    g = self.golden
    tfs = fp.compute_crop_window_tf(g["poses"], g["K"], 160, 160, 1.2, 0.1)
    np.testing.assert_allclose(tfs, g["tfs"], rtol=1e-6, atol=1e-5)

  def test_update_refined_poses(self):
    g = self.golden
    updated = fp.update_refined_poses(
        g["poses"], g["trans_delta"], g["rot_delta"], 0.1, 0.34906585
    )
    np.testing.assert_allclose(updated, g["updated"], atol=1e-6)

  def test_apply_mesh_center_offset(self):
    g = self.golden
    centered = fp.apply_mesh_center_offset(g["updated"][5], g["mesh_center"])
    np.testing.assert_allclose(centered, g["centered"], atol=1e-6)


class SamplingTest(parameterized.TestCase):

  @parameterized.parameters(1, 12, 40, 162)
  def test_rotation_grid_contains_rotations(self, n_views):
    poses = fp.make_rotation_grid(n_views, 60.0)
    # Accumulating the 60 degree step in floating point yields 7 in-plane
    # rotations, like the C++ loop.
    self.assertEqual(poses.shape, (n_views * 7, 4, 4))
    rotations = poses[:, :3, :3]
    np.testing.assert_allclose(
        rotations @ rotations.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1e-5,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-5)

  def test_empty_mask_uses_default_center(self):
    K = np.array([[500, 0, 32], [0, 500, 24], [0, 0, 1]], np.float32)
    poses = fp.sample_initial_poses(
        np.zeros((48, 64), np.uint8), np.ones((48, 64), np.float32), K, 1, 60.0
    )
    np.testing.assert_allclose(poses[:, :3, 3], [[0, 0, 0.8]] * len(poses))

  def test_translation_from_mask_center_and_median_depth(self):
    K = np.array([[500, 0, 32], [0, 500, 24], [0, 0, 1]], np.float32)
    mask = np.zeros((48, 64), np.uint8)
    mask[10:21, 40:51] = 1  # Center (u, v) = (45, 15).
    depth = np.full((48, 64), 2.0, np.float32)
    depth[10:21, 40:51] = 1.0
    depth[10, 40] = 0.05  # Below min_depth, ignored.
    poses = fp.sample_initial_poses(mask, depth, K, 1, 60.0)
    np.testing.assert_allclose(
        poses[0, :3, 3], [(45 - 32) / 500, (15 - 24) / 500, 1.0], atol=1e-6
    )


def _clip_vertices(xy_pixels, z_ndc, H, W):
  """Returns (1, V, 4) clip-space vertices for pixel positions and z/w."""
  xy = np.asarray(xy_pixels, np.float32)
  x = xy[:, 0] / W * 2.0 - 1.0
  y = xy[:, 1] / H * 2.0 - 1.0
  return np.stack([x, y, np.asarray(z_ndc, np.float32), np.ones_like(x)], -1)[
      None
  ]


class RasterizeCpuContextTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ctx = fp.RasterizeCpuContext(160, 160, 64)

  def test_square_coverage_and_triangle_ids(self):
    H, W = 8, 10
    # A square from pixel coordinates 2..6 x 1..5, covering the centers of
    # pixels x in [2, 5], y in [1, 4]. Rows are counted from y = -1.
    v = _clip_vertices([[2, 1], [6, 1], [6, 5], [2, 5]], [0.5] * 4, H, W)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    rast = self.ctx.rasterize(v, faces, H, W)
    self.assertEqual(rast.shape, (1, H, W, 4))
    expected = np.zeros((H, W), bool)
    expected[1:5, 2:6] = True
    np.testing.assert_array_equal(rast[0, :, :, 3] > 0, expected)
    self.assertEqual(set(np.unique(rast[0, :, :, 3])), {0.0, 1.0, 2.0})
    np.testing.assert_allclose(rast[0, expected, 2], 0.5)

  def test_interpolation_is_exact_for_linear_attributes(self):
    H, W = 16, 16
    v = _clip_vertices([[1, 1], [15, 2], [3, 14]], [0.1, 0.2, 0.3], H, W)
    faces = np.array([[0, 1, 2]], np.int32)
    rast = self.ctx.rasterize(v, faces, H, W)
    # Attributes that are linear in screen space: (x, y, 1).
    attrs = np.array([[[1, 1, 1], [15, 2, 1], [3, 14, 1]]], np.float32)
    out = self.ctx.interpolate(attrs, rast, faces, H, W)
    ys, xs = np.nonzero(rast[0, :, :, 3] > 0)
    self.assertGreater(len(xs), 50)
    np.testing.assert_allclose(out[0, ys, xs, 0], xs + 0.5, atol=1e-4)
    np.testing.assert_allclose(out[0, ys, xs, 1], ys + 0.5, atol=1e-4)
    np.testing.assert_allclose(out[0, ys, xs, 2], 1.0, atol=1e-5)
    np.testing.assert_array_equal(out[0, rast[0, :, :, 3] == 0], 0.0)

  def test_closest_triangle_wins_regardless_of_order(self):
    H, W = 6, 6
    corners = [[0, 0], [6, 0], [6, 6], [0, 6]]
    v = _clip_vertices(corners + corners, [0.2] * 4 + [0.1] * 4, H, W)
    faces = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.int32
    )
    for f in (faces, faces[::-1].copy()):
      rast = self.ctx.rasterize(v, f, H, W)
      ids = rast[0, :, :, 3].astype(int) - 1
      front = {i for i, face in enumerate(f) if face[0] >= 4}
      self.assertTrue(set(np.unique(ids)) <= front)
      np.testing.assert_allclose(rast[0, :, :, 2], 0.1)

  def test_fragments_outside_depth_range_are_clipped(self):
    H, W = 4, 4
    v = _clip_vertices([[0, 0], [4, 0], [4, 4], [0, 4]], [1.5] * 4, H, W)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    rast = self.ctx.rasterize(v, faces, H, W)
    self.assertFalse(np.any(rast[..., 3]))

  def test_batch_images_are_independent(self):
    H, W = 8, 8
    left = _clip_vertices([[0, 0], [4, 0], [4, 8], [0, 8]], [0.0] * 4, H, W)
    right = _clip_vertices([[4, 0], [8, 0], [8, 8], [4, 8]], [0.0] * 4, H, W)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    rast = self.ctx.rasterize(np.concatenate([left, right]), faces, H, W)
    self.assertTrue(np.all(rast[0, :, :4, 3] > 0))
    self.assertFalse(np.any(rast[0, :, 4:, 3]))
    self.assertTrue(np.all(rast[1, :, 4:, 3] > 0))
    self.assertFalse(np.any(rast[1, :, :4, 3]))

  def test_large_batch_is_split_into_chunks(self):
    H, W = 32, 32
    v = _clip_vertices([[0, 0], [32, 0], [32, 32], [0, 32]], [0.0] * 4, H, W)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    batch = np.repeat(v, 5, axis=0)
    with mock.patch.object(fp, "_MAX_FRAGMENTS_PER_CHUNK", 1000):
      chunked = self.ctx.rasterize(batch, faces, H, W)
    np.testing.assert_array_equal(
        chunked, self.ctx.rasterize(batch, faces, H, W)
    )
    self.assertTrue(np.all(chunked[..., 3] > 0))

  def test_invalid_faces_raise(self):
    v = _clip_vertices([[0, 0], [1, 0], [0, 1]], [0.0] * 3, 4, 4)
    with self.assertRaisesRegex(ValueError, "out of bounds"):
      self.ctx.rasterize(v, np.array([[0, 1, 3]], np.int32), 4, 4)


if __name__ == "__main__":
  absltest.main()
