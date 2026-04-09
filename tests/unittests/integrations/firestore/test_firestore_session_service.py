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
from google.adk.integrations.firestore.firestore_session_service import FirestoreSessionService
import pytest


@pytest.fixture
def mock_firestore_client():
  client = mock.MagicMock()
  collection_ref = mock.MagicMock()
  doc_ref = mock.MagicMock()
  subcollection_ref = mock.MagicMock()
  subdoc_ref = mock.MagicMock()

  client.collection.return_value = collection_ref
  collection_ref.document.return_value = doc_ref
  doc_ref.collection.return_value = subcollection_ref
  subcollection_ref.document.return_value = subdoc_ref

  doc_snapshot = mock.MagicMock()
  doc_snapshot.exists = False
  doc_snapshot.to_dict.return_value = {}

  doc_ref.get = mock.AsyncMock(return_value=doc_snapshot)
  subdoc_ref.get = mock.AsyncMock(return_value=doc_snapshot)

  subdoc_ref.set = mock.AsyncMock()
  subdoc_ref.delete = mock.AsyncMock()

  events_collection_ref = mock.MagicMock()
  subdoc_ref.collection.return_value = events_collection_ref
  events_collection_ref.order_by.return_value = events_collection_ref
  events_collection_ref.where.return_value = events_collection_ref
  events_collection_ref.limit_to_last.return_value = events_collection_ref
  events_collection_ref.get = mock.AsyncMock(return_value=[])

  subcollection_ref.get = mock.AsyncMock(return_value=[])
  subcollection_ref.where.return_value = subcollection_ref

  client.collection_group.return_value = collection_ref

  batch = mock.MagicMock()
  client.batch.return_value = batch
  batch.commit = mock.AsyncMock()

  return client


@pytest.mark.asyncio
async def test_create_session(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"

  session = await service.create_session(app_name=app_name, user_id=user_id)

  assert session.app_name == app_name
  assert session.user_id == user_id
  assert session.id

  mock_firestore_client.collection.assert_called_once_with("adk-session")
  collection_ref = mock_firestore_client.collection.return_value
  collection_ref.document.assert_called_once_with(user_id)
  doc_ref = collection_ref.document.return_value
  doc_ref.collection.assert_called_once_with("sessions")
  sessions_ref = doc_ref.collection.return_value
  sessions_ref.document.assert_called_once_with(session.id)
  session_doc_ref = sessions_ref.document.return_value
  session_doc_ref.set.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_not_found(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  session = await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert session is None


@pytest.mark.asyncio
async def test_get_session_found(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  doc_snapshot = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value
  )
  doc_snapshot.exists = True
  doc_snapshot.to_dict.return_value = {
      "id": session_id,
      "appName": app_name,
      "userId": user_id,
      "state": {"key": "value"},
      "updateTime": 1234567890.0,
  }

  session = await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert session is not None
  assert session.id == session_id
  assert session.state == {"key": "value"}


@pytest.mark.asyncio
async def test_delete_session(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  events_ref = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value
  )
  event_doc = mock.AsyncMock()
  events_ref.get = mock.AsyncMock(return_value=[event_doc])

  await service.delete_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  events_ref.get.assert_called_once()
  mock_firestore_client.batch.assert_called_once()
  batch = mock_firestore_client.batch.return_value
  batch.delete.assert_called_once_with(event_doc.reference)
  batch.commit.assert_called_once()

  session_doc_ref = (
      mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value
  )
  session_doc_ref.delete.assert_called_once()


@pytest.mark.asyncio
async def test_append_event(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name=app_name, user_id=user_id)
  event = Event(invocation_id="test_inv", author="user")

  await service.append_event(session, event)

  mock_firestore_client.batch.assert_called_once()
  batch = mock_firestore_client.batch.return_value
  batch.set.assert_called_once()
  batch.update.assert_called_once()
  batch.commit.assert_called_once()


@pytest.mark.asyncio
async def test_append_event_with_state_delta(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name=app_name, user_id=user_id)

  event = mock.MagicMock()
  event.partial = False
  event.id = "test_event_id"
  event.actions.state_delta = {
      "_app_my_key": "app_val",
      "_user_my_key": "user_val",
      "session_key": "session_val",
  }
  event.model_dump.return_value = {"id": "test_event_id", "author": "user"}

  await service.append_event(session, event)

  mock_firestore_client.batch.assert_called_once()
  batch = mock_firestore_client.batch.return_value

  batch.set.assert_called()

  assert session.state["session_key"] == "session_val"

  batch.update.assert_called_once()

  batch.commit.assert_called_once()
