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

from google.adk.events.event import Event
from google.adk.integrations.firestore.firestore_memory_service import FirestoreMemoryService
from google.genai import types
import pytest


@pytest.fixture
def mock_firestore_client():
  client = mock.MagicMock()
  collection_ref = mock.MagicMock()
  client.collection_group.return_value = collection_ref

  collection_ref.where.return_value = collection_ref

  doc_snapshot = mock.MagicMock()
  doc_snapshot.to_dict.return_value = {}

  collection_ref.get = mock.AsyncMock(return_value=[doc_snapshot])

  return client


def test_extract_keywords(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)
  text = "The quick brown fox jumps over the lazy dog."
  keywords = service._extract_keywords(text)

  assert "the" not in keywords
  assert "over" not in keywords
  assert "quick" in keywords
  assert "brown" in keywords
  assert "fox" in keywords
  assert "jumps" in keywords
  assert "lazy" in keywords
  assert "dog" in keywords


@pytest.mark.asyncio
async def test_search_memory_empty_query(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)
  response = await service.search_memory(
      app_name="test_app", user_id="test_user", query=""
  )
  assert not response.memories
  mock_firestore_client.collection_group.assert_not_called()


@pytest.mark.asyncio
async def test_search_memory_with_results(mock_firestore_client):
  service = FirestoreMemoryService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  query = "quick fox"

  doc_snapshot = mock_firestore_client.collection_group.return_value.where.return_value.where.return_value.where.return_value.get.return_value[
      0
  ]
  event = Event(
      invocation_id="test_inv",
      author="user",
      content=types.Content(parts=[types.Part(text="quick fox jumps")]),
  )
  doc_snapshot.to_dict.return_value = {
      "event_data": event.model_dump(exclude_none=True, mode="json")
  }

  response = await service.search_memory(
      app_name=app_name, user_id=user_id, query=query
  )

  assert response.memories
  assert len(response.memories) == 1
  assert response.memories[0].author == "user"

  mock_firestore_client.collection_group.assert_called_with("events")
  collection_ref = mock_firestore_client.collection_group.return_value
  collection_ref.where.assert_called()
