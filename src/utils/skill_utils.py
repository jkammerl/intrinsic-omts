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

"""Utilities for calling skills whose parameters differ across releases."""

import inspect
from typing import Any

# Camera slots of ai.intrinsic.estimate_pose_multi_view. Newer Intrinsic Core
# releases removed them; the skill then takes its images from capture_data only.
MULTI_VIEW_CAMERA_SLOTS = ("camera_1", "camera_2", "camera_3", "camera_4")


def multi_view_camera_kwargs(skill: Any, camera: Any) -> dict[str, Any]:
  """Returns the camera slot arguments the installed multi-view skill accepts.

  Args:
    skill: The generated estimate_pose_multi_view skill class.
    camera: Camera resource to fill every camera slot with.

  Returns:
    A dict mapping each camera slot the skill declares to `camera`; empty if
    the skill has no camera slots.
  """
  try:
    params = inspect.signature(skill).parameters
  except (TypeError, ValueError):
    return {}
  return {slot: camera for slot in MULTI_VIEW_CAMERA_SLOTS if slot in params}
