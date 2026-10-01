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

"""Tests FoundationPose's model.py without Triton and without a GPU.

The model runs from a Triton-style model directory (model.py next to the ONNX
files). Hermetic tests use small stand-in ONNX models for the refiner and the
scorer. The end-to-end test with the real FoundationPose weights runs if
FOUNDATIONPOSE_ONNX_DIR points to a directory containing
foundationpose_refine.onnx and foundationpose_score.onnx.
"""

import importlib.util
import json
import os
import shutil
import sys
import time
import types
from unittest import mock

from absl import logging
from absl.testing import absltest
# Imported before sys.modules is patched in the tests, so that model.py's
# imports of these modules are not undone after each test.
import cv2  # pylint: disable=unused-import
import numpy as np
import onnx
from onnx import helper
import onnxruntime  # pylint: disable=unused-import
from onnx import TensorProto
import trimesh

from third_party.foundationpose import foundationpose_numpy

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RAW_STOCK_GLB = os.path.join(
    _THIS_DIR, "..", "..", "models", "raw_stock_2x3x5", "base_visual.glb"
)
_ONNX_DIR_ENV = "FOUNDATIONPOSE_ONNX_DIR"
_DEVICE_ENV = "INTRINSIC_INFERENCE_DEVICE"

_K = np.array([[610.0, 0, 320.0], [0, 610.0, 240.0], [0, 0, 1]], np.float32)
_H, _W = 480, 640


def _fake_pb_utils() -> types.ModuleType:
  """Returns a minimal triton_python_backend_utils module."""
  pb_utils = types.ModuleType("triton_python_backend_utils")

  class Tensor:

    def __init__(self, name, array):
      self._name, self._array = name, np.asarray(array)

    def name(self):
      return self._name

    def as_numpy(self):
      return self._array

  class InferenceResponse:

    def __init__(self, output_tensors=(), error=None):
      self.output_tensors = {t.name(): t.as_numpy() for t in output_tensors}
      self.error = error

  class TritonError(Exception):
    pass

  def get_input_tensor_by_name(request, name):
    return Tensor(name, request[name]) if name in request else None

  pb_utils.Tensor = Tensor
  pb_utils.InferenceResponse = InferenceResponse
  pb_utils.TritonError = TritonError
  pb_utils.get_input_tensor_by_name = get_input_tensor_by_name
  return pb_utils


def _save(graph, path):
  onnx.save(
      helper.make_model(
          graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
      ),
      path,
  )


def _write_stand_in_onnx_models(model_dir):
  """Writes a refiner that predicts no change and a scorer rewarding overlap."""
  patch = ["batch_size", 160, 160, 6]
  inputs = [
      helper.make_tensor_value_info("input1", TensorProto.FLOAT, patch),
      helper.make_tensor_value_info("input2", TensorProto.FLOAT, patch),
  ]
  consts = [
      helper.make_tensor("starts", TensorProto.INT64, [3], [0, 0, 0]),
      helper.make_tensor("ends", TensorProto.INT64, [3], [1, 1, 3]),
      helper.make_tensor("axes", TensorProto.INT64, [3], [1, 2, 3]),
      helper.make_tensor("shape", TensorProto.INT64, [2], [-1, 3]),
      helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
  ]
  _save(
      helper.make_graph(
          [
              helper.make_node(
                  "Slice", ["input1", "starts", "ends", "axes"], ["s"]
              ),
              helper.make_node("Reshape", ["s", "shape"], ["r"]),
              helper.make_node("Mul", ["r", "zero"], ["output1"]),
              helper.make_node("Mul", ["r", "zero"], ["output2"]),
          ],
          "refine",
          inputs,
          [
              helper.make_tensor_value_info(
                  "output1", TensorProto.FLOAT, ["batch_size", 3]
              ),
              helper.make_tensor_value_info(
                  "output2", TensorProto.FLOAT, ["batch_size", 3]
              ),
          ],
          initializer=consts,
      ),
      os.path.join(model_dir, "foundationpose_refine.onnx"),
  )
  _save(
      helper.make_graph(
          [
              helper.make_node("Mul", ["input1", "input2"], ["prod"]),
              helper.make_node(
                  "ReduceMean", ["prod"], ["mean"], axes=[1, 2, 3], keepdims=0
              ),
              helper.make_node("Unsqueeze", ["mean", "axis0"], ["output1"]),
          ],
          "score",
          inputs,
          [
              helper.make_tensor_value_info(
                  "output1", TensorProto.FLOAT, [1, "batch_size"]
              )
          ],
          initializer=[
              helper.make_tensor("axis0", TensorProto.INT64, [1], [0]),
          ],
      ),
      os.path.join(model_dir, "foundationpose_score.onnx"),
  )


def _rotation(axis, angle_deg):
  axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
  return foundationpose_numpy._axis_angle_to_matrix(
      axis, np.deg2rad(angle_deg)
  )


def _render_scene(mesh, pose):
  """Renders RGB, depth and mask of `mesh` at `pose` (object in camera)."""
  verts = np.asarray(mesh.vertices, np.float32)
  faces = np.asarray(mesh.faces, np.int32)
  v_cam = verts @ pose[:3, :3].T + pose[:3, 3]
  z = v_cam[:, 2]
  px = v_cam[:, 0] * _K[0, 0] / z + _K[0, 2]
  py = v_cam[:, 1] * _K[1, 1] / z + _K[1, 2]
  ndc = np.stack(
      [2 * px / _W - 1, 2 * py / _H - 1, (z - 0.1) / (100.0 - 0.1)], -1
  )
  # Homogeneous clip coordinates with w = z for perspective-correct depth.
  v_clip = np.concatenate([ndc * z[:, None], z[:, None]], -1)[None]
  ctx = foundationpose_numpy.RasterizeCpuContext()
  rast = ctx.rasterize(v_clip.astype(np.float32), faces, _H, _W)
  xyz = ctx.interpolate(v_cam[None].astype(np.float32), rast, faces, _H, _W)
  mask = rast[0, :, :, 3] > 0

  # Table plane 5 cm behind the object's center, and Lambertian shading.
  depth = np.full((_H, _W), pose[2, 3] + 0.05, np.float32)
  depth[mask] = xyz[0, mask, 2]
  face_normals = np.asarray(mesh.face_normals) @ pose[:3, :3].T
  light = np.array([0.3, -0.5, -1.0]) / np.linalg.norm([0.3, -0.5, -1.0])
  shade = np.clip(face_normals @ light, 0.15, 1.0)
  rgb = np.full((_H, _W, 3), 90, np.uint8)
  face_ids = rast[0, :, :, 3].astype(np.int64) - 1
  rgb[mask] = (shade[face_ids[mask]][:, None] * [200, 205, 210]).astype(
      np.uint8
  )
  return rgb, depth, mask.astype(np.uint8)


def _add_s(mesh, pose_a, pose_b, samples=2000):
  """Symmetric average distance (ADD-S) between two poses of `mesh`."""
  points = trimesh.sample.sample_surface(mesh, samples, seed=0)[0]
  a = points @ pose_a[:3, :3].T + pose_a[:3, 3]
  b = points @ pose_b[:3, :3].T + pose_b[:3, 3]
  distances = np.linalg.norm(a[:, None] - b[None], axis=-1)
  return distances.min(axis=1).mean()


class _ModelTestBase(absltest.TestCase):
  """Loads model.py from a Triton-style model version directory."""

  def setUp(self):
    super().setUp()
    self.model_dir = self.create_tempdir().full_path
    for name in ("model.py", "foundationpose_numpy.py"):
      shutil.copy(os.path.join(_THIS_DIR, name), self.model_dir)
    self.enter_context(
        mock.patch.dict(
            sys.modules, {"triton_python_backend_utils": _fake_pb_utils()}
        )
    )
    self.enter_context(mock.patch.object(sys, "path", list(sys.path)))
    with open(_RAW_STOCK_GLB, "rb") as f:
      self.cad_bytes = f.read()
    self.mesh = trimesh.load(_RAW_STOCK_GLB, file_type="glb", force="mesh")

  def load_model(self, device=None, foundationpose_cpp=None):
    env = {} if device is None else {_DEVICE_ENV: device}
    modules = {}
    if foundationpose_cpp is not None:
      modules["foundationpose_cpp"] = foundationpose_cpp
    with mock.patch.dict(os.environ, env), mock.patch.dict(
        sys.modules, modules
    ):
      spec = importlib.util.spec_from_file_location(
          "foundationpose_model_under_test",
          os.path.join(self.model_dir, "model.py"),
      )
      module = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(module)
      model = module.TritonPythonModel()
      model.initialize({"model_config": json.dumps({})})
    return model

  def execute(self, model, request):
    (response,) = model.execute([request])
    return response


class ModelTest(_ModelTestBase):

  def setUp(self):
    super().setUp()
    _write_stand_in_onnx_models(self.model_dir)
    self.pose = np.eye(4, dtype=np.float32)
    self.pose[:3, :3] = _rotation([1, 0.2, 0], 35)
    self.pose[:3, 3] = [0.03, -0.02, 0.55]
    self.rgb, self.depth, self.mask = _render_scene(self.mesh, self.pose)

  def _request(self, masks, **extra):
    request = {
        "RGB": self.rgb,
        "DEPTH": self.depth,
        "MASK": masks,
        "CAM_K": _K,
        "CAD_MODEL_BYTES": np.frombuffer(self.cad_bytes, np.uint8),
        "NUM_ITERATIONS": np.array([1], np.int32),
    }
    request.update(extra)
    return request

  def test_cpu_device_uses_cpu_providers_and_rasterizer(self):
    model = self.load_model("cpu")
    self.assertEqual(
        model.refine_session.get_providers(), ["CPUExecutionProvider"]
    )
    # model.py uses its own copy of foundationpose_numpy from the model dir.
    self.assertEqual(type(model.glctx).__name__, "RasterizeCpuContext")
    self.assertIs(model.helpers.__name__, "foundationpose_numpy")

  def test_gpu_device_without_gpu_fails(self):
    with mock.patch(
        "onnxruntime.get_available_providers",
        return_value=["CPUExecutionProvider"],
    ):
      with self.assertRaisesRegex(RuntimeError, "no GPU execution provider"):
        self.load_model("gpu")

  def test_invalid_device_fails(self):
    with self.assertRaisesRegex(ValueError, "INTRINSIC_INFERENCE_DEVICE"):
      self.load_model("tpu")

  def test_uses_cpp_helpers_and_falls_back_to_cpu_rasterizer(self):
    # foundationpose_cpp is installed, but no CUDA device is available.
    fake_cpp = types.ModuleType("foundationpose_cpp")
    for name in (
        "sample_initial_poses",
        "compute_crop_window_tf",
        "update_refined_poses",
        "apply_mesh_center_offset",
    ):
      setattr(
          fake_cpp,
          name,
          mock.Mock(wraps=getattr(foundationpose_numpy, name)),
      )
    fake_cpp.RasterizeCudaContext = mock.Mock(
        side_effect=RuntimeError("no CUDA device")
    )
    model = self.load_model("auto", foundationpose_cpp=fake_cpp)
    self.assertIs(model.helpers, fake_cpp)
    # model.py uses its own copy of foundationpose_numpy from the model dir.
    self.assertEqual(type(model.glctx).__name__, "RasterizeCpuContext")
    response = self.execute(model, self._request(self.mask[None]))
    self.assertIsNone(response.error)
    fake_cpp.sample_initial_poses.assert_called_once()
    fake_cpp.update_refined_poses.assert_called()

  def test_estimates_one_pose_per_mask(self):
    model = self.load_model("cpu")
    masks = np.stack([self.mask, self.mask])
    response = self.execute(model, self._request(masks))
    self.assertIsNone(response.error)
    rotations = response.output_tensors["ROTATION"]
    translations = response.output_tensors["TRANSLATION"]
    confidences = response.output_tensors["CONFIDENCE"]
    self.assertEqual(rotations.shape, (2, 3, 3))
    self.assertEqual(translations.shape, (2, 3))
    self.assertEqual(confidences.shape, (2, 1))
    for array in (rotations, translations, confidences):
      self.assertEqual(array.dtype, np.float32)
    np.testing.assert_allclose(
        rotations @ rotations.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        atol=1e-4,
    )
    # Without refinement, the translation is the initial guess from the mask
    # and depth, which is close to the object's center.
    np.testing.assert_allclose(translations[0], self.pose[:3, 3], atol=0.03)
    np.testing.assert_array_equal(translations[0], translations[1])

  def test_non_finite_depth_is_treated_as_invalid(self):
    # Simulated depth cameras report inf (or NaN) where nothing is hit.
    # They must give the same result as depth 0, FoundationPose's invalid value.
    model = self.load_model("cpu")
    outside = self.mask == 0
    zero_depth = self.depth.copy()
    zero_depth[outside] = 0
    expected = self.execute(
        model, self._request(self.mask[None], DEPTH=zero_depth)
    )
    depth = self.depth.copy()
    depth[outside] = np.inf
    depth[0, 0] = np.nan
    response = self.execute(
        model, self._request(self.mask[None], DEPTH=depth)
    )
    self.assertIsNone(response.error)
    for name, array in response.output_tensors.items():
      self.assertTrue(np.isfinite(array).all(), name)
      np.testing.assert_array_equal(array, expected.output_tensors[name])

  def test_batch_size_does_not_change_result(self):
    model = self.load_model("cpu")
    full = self.execute(model, self._request(self.mask[None]))
    batched = self.execute(
        model,
        self._request(self.mask[None], BATCH_SIZE=np.array([50], np.int32)),
    )
    for name in ("ROTATION", "TRANSLATION", "CONFIDENCE"):
      np.testing.assert_allclose(
          full.output_tensors[name], batched.output_tensors[name], atol=1e-5
      )

  def test_no_masks_returns_empty_outputs(self):
    model = self.load_model("cpu")
    response = self.execute(
        model, self._request(np.zeros((0, _H, _W), np.uint8))
    )
    self.assertIsNone(response.error)
    self.assertEqual(response.output_tensors["ROTATION"].shape, (0, 3, 3))
    self.assertEqual(response.output_tensors["CONFIDENCE"].shape, (0, 1))

  def test_invalid_cad_model_returns_error(self):
    model = self.load_model("cpu")
    request = self._request(self.mask[None])
    request["CAD_MODEL_BYTES"] = np.frombuffer(b"not a mesh", np.uint8)
    response = self.execute(model, request)
    self.assertIsNotNone(response.error)
    self.assertIn("no valid vertices", str(response.error))


@absltest.skipUnless(
    os.environ.get(_ONNX_DIR_ENV), f"{_ONNX_DIR_ENV} is not set"
)
class RealWeightsTest(_ModelTestBase):
  """Runs the real FoundationPose networks on a rendered scene."""

  def setUp(self):
    super().setUp()
    onnx_dir = os.environ[_ONNX_DIR_ENV]
    for name in ("foundationpose_refine.onnx", "foundationpose_score.onnx"):
      os.symlink(
          os.path.join(onnx_dir, name), os.path.join(self.model_dir, name)
      )

  def test_recovers_pose_of_raw_stock(self):
    model = self.load_model("auto")
    diameter = float(np.linalg.norm(self.mesh.extents))
    # The block lying on a table, seen from a camera about 0.6 m away.
    for axis, angle, translation in [
        ([1, 0, 0], 140, [0.02, 0.01, 0.6]),
        ([1, 0.3, 0.2], 125, [-0.05, 0.04, 0.55]),
    ]:
      pose = np.eye(4, dtype=np.float32)
      pose[:3, :3] = _rotation(axis, angle)
      pose[:3, 3] = translation
      rgb, depth, mask = _render_scene(self.mesh, pose)
      request = {
          "RGB": rgb,
          "DEPTH": depth,
          "MASK": mask[None],
          "CAM_K": _K,
          "CAD_MODEL_BYTES": np.frombuffer(self.cad_bytes, np.uint8),
          "NUM_ITERATIONS": np.array([3], np.int32),
      }
      start = time.perf_counter()
      response = self.execute(model, request)
      logging.info(
          "FoundationPose on %s took %.1f s",
          model.refine_session.get_providers()[0],
          time.perf_counter() - start,
      )
      self.assertIsNone(response.error)
      estimate = np.eye(4, dtype=np.float32)
      estimate[:3, :3] = response.output_tensors["ROTATION"][0]
      estimate[:3, 3] = response.output_tensors["TRANSLATION"][0]
      error = _add_s(self.mesh, estimate, pose)
      logging.info(
          "ADD-S error %.1f mm (%.1f%% of the diameter), translation error"
          " %.1f mm",
          error * 1000,
          100 * error / diameter,
          1000 * np.linalg.norm(estimate[:3, 3] - pose[:3, 3]),
      )
      # The BOP/YCB-Video standard for a correct pose: ADD-S below 10% of
      # the object's diameter.
      self.assertLess(error, 0.1 * diameter)


if __name__ == "__main__":
  absltest.main()
