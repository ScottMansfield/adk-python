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

  async def to_async_iter(iterable):
    for item in iterable:
      yield item

  events_ref.stream.return_value = to_async_iter([event_doc])

  await service.delete_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  events_ref.stream.assert_called_once()
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

  service._update_app_state_transactional = mock.AsyncMock()
  service._update_user_state_transactional = mock.AsyncMock()

  await service.append_event(session, event)

  mock_firestore_client.batch.assert_called_once()
  service._update_app_state_transactional.assert_called_once_with(
      "test_app", {"my_key": "app_val"}
  )
  service._update_user_state_transactional.assert_called_once_with(
      "test_app", "test_user", {"my_key": "user_val"}
  )

  batch = mock_firestore_client.batch.return_value

  batch.set.assert_called()

  assert session.state["session_key"] == "session_val"

  batch.update.assert_called_once()

  batch.commit.assert_called_once()


@pytest.mark.asyncio
async def test_list_sessions_with_user_id(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"

  session_doc = mock.MagicMock()
  session_doc.to_dict.return_value = {
      "id": "session1",
      "appName": app_name,
      "userId": user_id,
      "state": {"session_key": "session_val"},
  }

  app_state_coll = mock.MagicMock()
  user_state_coll = mock.MagicMock()
  sessions_coll = mock.MagicMock()

  def collection_side_effect(name):
    if name == service.app_state_collection:
      return app_state_coll
    elif name == service.user_state_collection:
      return user_state_coll
    elif name == service.root_collection:
      return sessions_coll
    return mock.MagicMock()

  mock_firestore_client.collection.side_effect = collection_side_effect

  app_doc = mock.MagicMock()
  app_doc.exists = True
  app_doc.to_dict.return_value = {"app_key": "app_val"}
  app_doc_ref = mock.MagicMock()
  app_state_coll.document.return_value = app_doc_ref
  app_doc_ref.get = mock.AsyncMock(return_value=app_doc)

  user_doc = mock.MagicMock()
  user_doc.exists = True
  user_doc.to_dict.return_value = {"user_key": "user_val"}
  user_app_doc = mock.MagicMock()
  user_state_coll.document.return_value = user_app_doc
  users_coll = mock.MagicMock()
  user_app_doc.collection.return_value = users_coll
  user_doc_ref = mock.MagicMock()
  users_coll.document.return_value = user_doc_ref
  user_doc_ref.get = mock.AsyncMock(return_value=user_doc)

  user_doc_in_sessions = mock.MagicMock()
  sessions_coll.document.return_value = user_doc_in_sessions
  sessions_subcoll = mock.MagicMock()
  user_doc_in_sessions.collection.return_value = sessions_subcoll
  sessions_query = mock.MagicMock()
  sessions_subcoll.where.return_value = sessions_query
  sessions_query.get = mock.AsyncMock(return_value=[session_doc])

  response = await service.list_sessions(app_name=app_name, user_id=user_id)

  assert len(response.sessions) == 1
  session = response.sessions[0]
  assert session.id == "session1"
  assert session.state["session_key"] == "session_val"
  assert session.state["_app_app_key"] == "app_val"
  assert session.state["_user_user_key"] == "user_val"


@pytest.mark.asyncio
async def test_list_sessions_without_user_id(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"

  session_doc = mock.MagicMock()
  session_doc.to_dict.return_value = {
      "id": "session1",
      "appName": app_name,
      "userId": "user1",
      "state": {"session_key": "session_val"},
  }

  mock_firestore_client.collection_group.return_value.where.return_value.get = mock.AsyncMock(
      return_value=[session_doc]
  )

  app_state_coll = mock.MagicMock()
  user_state_coll = mock.MagicMock()

  def collection_side_effect(name):
    if name == service.app_state_collection:
      return app_state_coll
    elif name == service.user_state_collection:
      return user_state_coll
    return mock.MagicMock()

  mock_firestore_client.collection.side_effect = collection_side_effect

  app_doc = mock.MagicMock()
  app_doc.exists = True
  app_doc.to_dict.return_value = {"app_key": "app_val"}
  app_doc_ref = mock.MagicMock()
  app_state_coll.document.return_value = app_doc_ref
  app_doc_ref.get = mock.AsyncMock(return_value=app_doc)

  user_doc = mock.MagicMock()
  user_doc.id = "user1"
  user_doc.to_dict.return_value = {"user_key": "user_val"}
  user_app_doc = mock.MagicMock()
  user_state_coll.document.return_value = user_app_doc
  users_coll = mock.MagicMock()
  user_app_doc.collection.return_value = users_coll
  users_coll.get = mock.AsyncMock(return_value=[user_doc])

  response = await service.list_sessions(app_name=app_name)

  assert len(response.sessions) == 1
  session = response.sessions[0]
  assert session.id == "session1"
  assert session.state["_app_app_key"] == "app_val"
  assert session.state["_user_user_key"] == "user_val"


@pytest.mark.asyncio
async def test_create_session_already_exists(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"

  doc_snapshot = mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value
  doc_snapshot.exists = True

  from google.adk.errors.already_exists_error import AlreadyExistsError

  with pytest.raises(AlreadyExistsError):
    await service.create_session(
        app_name=app_name, user_id=user_id, session_id="existing_id"
    )


@pytest.mark.asyncio
async def test_get_session_with_config(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  doc_snapshot = mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.get.return_value
  doc_snapshot.exists = True
  doc_snapshot.to_dict.return_value = {
      "id": session_id,
      "appName": app_name,
      "userId": user_id,
  }

  events_collection_ref = mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value

  from google.adk.sessions.base_session_service import GetSessionConfig

  config = GetSessionConfig(after_timestamp=1234567890.0, num_recent_events=5)

  await service.get_session(
      app_name=app_name, user_id=user_id, session_id=session_id, config=config
  )

  events_collection_ref.where.assert_called_once()
  events_collection_ref.limit_to_last.assert_called_once_with(5)


@pytest.mark.asyncio
async def test_delete_session_batching(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  app_name = "test_app"
  user_id = "test_user"
  session_id = "test_session"

  events_ref = mock_firestore_client.collection.return_value.document.return_value.collection.return_value.document.return_value.collection.return_value

  dummy_docs = [mock.MagicMock() for _ in range(501)]

  async def to_async_iter(iterable):
    for item in iterable:
      yield item

  events_ref.stream.return_value = to_async_iter(dummy_docs)

  batch = mock_firestore_client.batch.return_value

  await service.delete_session(
      app_name=app_name, user_id=user_id, session_id=session_id
  )

  assert batch.commit.call_count == 2


@pytest.mark.asyncio
async def test_append_event_partial(mock_firestore_client):
  service = FirestoreSessionService(client=mock_firestore_client)
  from google.adk.sessions.session import Session

  session = Session(id="test_session", app_name="test_app", user_id="test_user")

  event = Event(invocation_id="test_inv", author="user", partial=True)

  result = await service.append_event(session, event)

  assert result == event
  mock_firestore_client.batch.assert_not_called()

