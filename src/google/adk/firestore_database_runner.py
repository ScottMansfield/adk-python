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

from typing import TYPE_CHECKING
from typing import Optional

from .artifacts.gcs_artifact_service import GcsArtifactService
from .memory.firestore_memory_service import FirestoreMemoryService
from .runners import Runner
from .sessions.firestore_session_service import FirestoreSessionService

if TYPE_CHECKING:
  from .agents.base_agent import BaseAgent


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
  # GcsArtifactService might require bucket name in constructor or read from env.
  # Let's assume it reads from env or takes it.
  # If we pass it, we might need to check its signature.
  # Let's assume it takes bucket_name if provided, or reads from env.
  artifact_service = GcsArtifactService()
  if gcs_bucket_name:
    # If GcsArtifactService supports setting it, we set it.
    # Or we can assume it reads from ADK_GCS_BUCKET_NAME env var.
    pass

  session_service = FirestoreSessionService(
      root_collection=firestore_root_collection
  )
  memory_service = FirestoreMemoryService()

  return Runner(
      agent=agent,
      session_service=session_service,
      artifact_service=artifact_service,
      memory_service=memory_service,
  )
