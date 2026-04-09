# Copyright 2026 Google LLC
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

from __future__ import annotations

import os
from typing import Optional
from typing import TYPE_CHECKING

from ...artifacts.gcs_artifact_service import GcsArtifactService
from ...runners import Runner
from .firestore_memory_service import FirestoreMemoryService
from .firestore_session_service import FirestoreSessionService

if TYPE_CHECKING:
  from ...agents.base_agent import BaseAgent


def create_firestore_runner(
    agent: BaseAgent,
    gcs_bucket_name: Optional[str] = None,
    firestore_root_collection: Optional[str] = None,
) -> Runner:
  """Creates a Runner configured with Firestore and GCS services.

  Args:
    agent: The root agent to run.
    gcs_bucket_name: The GCS bucket name for artifacts.
    firestore_root_collection: The root collection name for Firestore.

  Returns:
    A Runner instance configured with Firestore services.
  """
  bucket_name = gcs_bucket_name or os.environ.get("ADK_GCS_BUCKET_NAME")
  if not bucket_name:
    raise ValueError(
        "Required property 'ADK_GCS_BUCKET_NAME' is not set. This"
        " is needed for the GcsArtifactService."
    )
  artifact_service = GcsArtifactService(bucket_name=bucket_name)

  session_service = FirestoreSessionService(
      root_collection=firestore_root_collection
  )
  memory_service = FirestoreMemoryService()

  return Runner(
      app_name=agent.name,
      agent=agent,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
  )
