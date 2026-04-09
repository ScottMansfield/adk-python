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

from unittest import mock

from google.adk.agents.base_agent import BaseAgent
from google.adk.integrations.firestore.firestore_database_runner import create_firestore_runner
import pytest


@pytest.fixture
def mock_agent():
  agent = mock.MagicMock(spec=BaseAgent)
  agent.name = "test_agent"
  return agent


def test_create_firestore_runner_with_arg(mock_agent, monkeypatch):
  monkeypatch.delenv("ADK_GCS_BUCKET_NAME", raising=False)

  with (
      mock.patch(
          "google.adk.integrations.firestore.firestore_database_runner.FirestoreSessionService"
      ),
      mock.patch("google.adk.integrations.firestore.firestore_database_runner.FirestoreMemoryService"),
      mock.patch(
          "google.adk.integrations.firestore.firestore_database_runner.GcsArtifactService"
      ) as mock_gcs,
  ):
    runner = create_firestore_runner(mock_agent, gcs_bucket_name="test_bucket")

    assert runner is not None
    mock_gcs.assert_called_once_with(bucket_name="test_bucket")


def test_create_firestore_runner_with_env(mock_agent, monkeypatch):
  monkeypatch.setenv("ADK_GCS_BUCKET_NAME", "env_bucket")

  with (
      mock.patch(
          "google.adk.integrations.firestore.firestore_database_runner.FirestoreSessionService"
      ),
      mock.patch("google.adk.integrations.firestore.firestore_database_runner.FirestoreMemoryService"),
      mock.patch(
          "google.adk.integrations.firestore.firestore_database_runner.GcsArtifactService"
      ) as mock_gcs,
  ):
    runner = create_firestore_runner(mock_agent)

    assert runner is not None
    mock_gcs.assert_called_once_with(bucket_name="env_bucket")


def test_create_firestore_runner_missing_bucket(mock_agent, monkeypatch):
  monkeypatch.delenv("ADK_GCS_BUCKET_NAME", raising=False)

  with pytest.raises(
      ValueError, match="Required property 'ADK_GCS_BUCKET_NAME' is not set"
  ):
    create_firestore_runner(mock_agent)


def test_create_firestore_runner_with_root_collection(mock_agent, monkeypatch):
  monkeypatch.setenv("ADK_GCS_BUCKET_NAME", "test_bucket")

  with (
      mock.patch(
          "google.adk.integrations.firestore.firestore_database_runner.FirestoreSessionService"
      ) as mock_session,
      mock.patch("google.adk.integrations.firestore.firestore_database_runner.FirestoreMemoryService"),
      mock.patch("google.adk.integrations.firestore.firestore_database_runner.GcsArtifactService"),
  ):
    runner = create_firestore_runner(
        mock_agent, firestore_root_collection="custom_collection"
    )

    assert runner is not None
    mock_session.assert_called_once_with(root_collection="custom_collection")

