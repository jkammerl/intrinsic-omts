# Copyright 2026 Intrinsic Innovation LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for skill_utils module."""

import inspect

from absl.testing import absltest

from src.utils import skill_utils


def _skill_with_params(*names: str):
  """Returns a callable whose signature lists `names`, like generated skills."""

  def skill(**kwargs):
    return kwargs

  skill.__signature__ = inspect.Signature(
    [inspect.Parameter(n, inspect.Parameter.KEYWORD_ONLY) for n in names]
  )
  return skill


class SkillUtilsTest(absltest.TestCase):
  def test_fills_camera_slots_when_skill_declares_them(self):
    skill = _skill_with_params(
      "camera_1", "camera_2", "camera_3", "camera_4", "perception"
    )
    self.assertEqual(
      skill_utils.multi_view_camera_kwargs(skill, "cam"),
      {
        "camera_1": "cam",
        "camera_2": "cam",
        "camera_3": "cam",
        "camera_4": "cam",
      },
    )

  def test_omits_camera_slots_when_skill_has_none(self):
    skill = _skill_with_params("perception", "capture_data")
    self.assertEqual(skill_utils.multi_view_camera_kwargs(skill, "cam"), {})

  def test_returns_empty_for_objects_without_signature(self):
    self.assertEqual(skill_utils.multi_view_camera_kwargs(42, "cam"), {})


if __name__ == "__main__":
  absltest.main()
