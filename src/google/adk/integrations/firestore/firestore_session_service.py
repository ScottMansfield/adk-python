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

from datetime import datetime
from datetime import timezone
import logging
import os
from typing import Any
from typing import Optional

from pydantic import BaseModel

from ...events.event import Event
from ...sessions.base_session_service import BaseSessionService
from ...sessions.base_session_service import GetSessionConfig
from ...sessions.base_session_service import ListSessionsResponse
from ...sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_ROOT_COLLECTION = "adk-session"
DEFAULT_SESSIONS_COLLECTION = "sessions"
DEFAULT_EVENTS_COLLECTION = "events"
DEFAULT_APP_STATE_COLLECTION = "app_states"
DEFAULT_USER_STATE_COLLECTION = "user_states"


class FirestoreSessionService(BaseSessionService):
  """Session service that uses Google Cloud Firestore as the backend."""

  def __init__(
      self,
      client: Optional[firestore.AsyncClient] = None,
      root_collection: Optional[str] = None,
  ):
    """Initializes the Firestore session service.

    Args:
      client: An optional Firestore AsyncClient. If not provided, a new one
        will be created.
      root_collection: The root collection name. Defaults to 'adk-session' or
        the value of ADK_FIRESTORE_ROOT_COLLECTION env var.
    """
    try:
      from google.cloud import firestore
    except ImportError as e:
      raise ImportError(
          "FirestoreSessionService requires google-cloud-firestore. "
          "Install it with: pip install google-cloud-firestore"
      ) from e

    self.client = client or firestore.AsyncClient()
    self.root_collection = (
         root_collection
        or os.environ.get("ADK_FIRESTORE_ROOT_COLLECTION")
        or DEFAULT_ROOT_COLLECTION
    )
    self.sessions_collection = DEFAULT_SESSIONS_COLLECTION
    self.events_collection = DEFAULT_EVENTS_COLLECTION
    self.app_state_collection = DEFAULT_APP_STATE_COLLECTION
    self.user_state_collection = DEFAULT_USER_STATE_COLLECTION

  @staticmethod
  def _merge_state(
      app_state: dict[str, Any],
      user_state: dict[str, Any],
      session_state: dict[str, Any],
  ) -> dict[str, Any]:
    """Merge app, user, and session states into a single state dictionary."""
    import copy

    merged_state = copy.deepcopy(session_state)
    for key, value in app_state.items():
      merged_state["_app_" + key] = value
    for key, value in user_state.items():
      merged_state["_user_" + key] = value
    return merged_state

  def _get_sessions_ref(
      self, user_id: str
  ) -> firestore.AsyncCollectionReference:
    return (
        self.client.collection(self.root_collection)
        .document(user_id)
        .collection(self.sessions_collection)
    )

  async def create_session(
      self,
      *,
      app_name: str,
      user_id: str,
      state: Optional[dict[str, Any]] = None,
      session_id: Optional[str] = None,
  ) -> Session:
    """Creates a new session in Firestore."""
    from google.cloud import firestore
    if not session_id:
      from ...platform import uuid as platform_uuid

      session_id = platform_uuid.new_uuid()

    initial_state = state or {}
    now = firestore.SERVER_TIMESTAMP

    session_ref = self._get_sessions_ref(user_id).document(session_id)

    # Check if session already exists
    doc = await session_ref.get()
    if doc.exists:
      from ...errors.already_exists_error import AlreadyExistsError

      raise AlreadyExistsError(f"Session {session_id} already exists.")

    session_data = {
        "id": session_id,
        "appName": app_name,
        "userId": user_id,
        "state": initial_state,
        "createTime": now,
        "updateTime": now,
    }

    await session_ref.set(session_data)

    # We need a timestamp for the Session object. Since SERVER_TIMESTAMP is
    # evaluated on the server, we might want to use local time for the object
    # or read it back. Reading it back is expensive. We'll use local time for
    # the object, but the DB will have SERVER_TIMESTAMP.
    local_now = datetime.now(timezone.utc).timestamp()

    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=initial_state,
        events=[],
        last_update_time=local_now,
    )

  async def get_session(
      self,
      *,
      app_name: str,
      user_id: str,
      session_id: str,
      config: Optional[GetSessionConfig] = None,
  ) -> Optional[Session]:
    """Gets a session from Firestore."""
    session_ref = self._get_sessions_ref(user_id).document(session_id)
    doc = await session_ref.get()

    if not doc.exists:
      return None

    data = doc.to_dict()
    if not data:
      return None

    # Fetch events
    events_ref = session_ref.collection(self.events_collection)
    query = events_ref.order_by("timestamp")

    if config:
      if config.after_timestamp:
        after_dt = datetime.fromtimestamp(config.after_timestamp)
        query = query.where("timestamp", ">=", after_dt)
      if config.num_recent_events:
        query = query.limit_to_last(config.num_recent_events)

    events_docs = await query.get()
    events = []
    for event_doc in events_docs:
      event_data = event_doc.to_dict()
      if event_data and "event_data" in event_data:
        ed = event_data["event_data"]
        events.append(Event.model_validate(ed))

    # Let's continue getting session.
    session_state = data.get("state", {})

    # Convert timestamp
    update_time = data.get("updateTime")
    last_update_time = 0.0
    if update_time:
      if isinstance(update_time, datetime):
        last_update_time = update_time.timestamp()
      else:
        try:
          last_update_time = float(update_time)
        except (ValueError, TypeError):
          pass

    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=session_state,
        events=events,
        last_update_time=last_update_time,
    )

  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    """Lists sessions from Firestore."""
    if user_id:
      query = self._get_sessions_ref(user_id).where("appName", "==", app_name)
      docs = await query.get()
    else:
      query = self.client.collection_group(self.sessions_collection).where(
          "appName", "==", app_name
      )
      docs = await query.get()

    # Fetch shared state once
    app_ref = self.client.collection(self.app_state_collection).document(
        app_name
    )
    app_doc = await app_ref.get()
    app_state = app_doc.to_dict() if app_doc.exists else {}

    user_states_map = {}
    if user_id:
      user_ref = (
          self.client.collection(self.user_state_collection)
          .document(app_name)
          .collection("users")
          .document(user_id)
      )
      user_doc = await user_ref.get()
      if user_doc.exists:
        user_states_map[user_id] = user_doc.to_dict()
    else:
      users_ref = (
          self.client.collection(self.user_state_collection)
          .document(app_name)
          .collection("users")
      )
      users_docs = await users_ref.get()
      for u_doc in users_docs:
        user_states_map[u_doc.id] = u_doc.to_dict()

    sessions = []
    for doc in docs:
      data = doc.to_dict()
      if data:
        u_id = data["userId"]
        s_state = data.get("state", {})
        u_state = user_states_map.get(u_id, {})
        merged = self._merge_state(app_state, u_state, s_state)

        sessions.append(
            Session(
                id=data["id"],
                app_name=data["appName"],
                user_id=data["userId"],
                state=merged,
                events=[],
                last_update_time=0.0,
            )
        )

    return ListSessionsResponse(sessions=sessions)

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Deletes a session and its events from Firestore."""
    session_ref = self._get_sessions_ref(user_id).document(session_id)

    events_ref = session_ref.collection(self.events_collection)
    
    batch = self.client.batch()
    count = 0
    async for event_doc in events_ref.stream():
      batch.delete(event_doc.reference)
      count += 1
      if count >= 500:
        await batch.commit()
        batch = self.client.batch()
        count = 0
    if count > 0:
      await batch.commit()

    await session_ref.delete()

  async def _update_app_state_transactional(
      self, app_name: str, delta: dict[str, Any]
  ) -> dict[str, Any]:
    """Atomically applies delta to app state inside a transaction."""
    from google.cloud import firestore
    doc_ref = self.client.collection(self.app_state_collection).document(app_name)

    @firestore.async_transactional
    async def _txn(transaction):
      snap = await doc_ref.get(transaction=transaction)
      current = snap.to_dict() if snap.exists else {}
      current.update(delta)
      transaction.set(doc_ref, current, merge=True)
      return current

    transaction = self.client.transaction()
    return await _txn(transaction)

  async def _update_user_state_transactional(
      self, app_name: str, user_id: str, delta: dict[str, Any]
  ) -> dict[str, Any]:
    """Atomically applies delta to user state inside a transaction."""
    from google.cloud import firestore
    doc_ref = (
        self.client.collection(self.user_state_collection)
        .document(app_name)
        .collection("users")
        .document(user_id)
    )

    @firestore.async_transactional
    async def _txn(transaction):
      snap = await doc_ref.get(transaction=transaction)
      current = snap.to_dict() if snap.exists else {}
      current.update(delta)
      transaction.set(doc_ref, current, merge=True)
      return current

    transaction = self.client.transaction()
    return await _txn(transaction)

  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event to a session in Firestore."""
    from google.cloud import firestore
    if event.partial:
      return event

    self._apply_temp_state(session, event)
    event = self._trim_temp_delta_state(event)

    session_ref = self._get_sessions_ref(session.user_id).document(session.id)

    if event.actions and event.actions.state_delta:
      state_delta = event.actions.state_delta
      app_updates = {}
      user_updates = {}
      session_updates = {}

      for key, value in state_delta.items():
        if key.startswith("_app_"):
          app_updates[key[len("_app_") :]] = value
        elif key.startswith("_user_"):
          user_updates[key[len("_user_") :]] = value
        else:
          session_updates[key] = value

      if app_updates:
        await self._update_app_state_transactional(session.app_name, app_updates)

      if user_updates:
        await self._update_user_state_transactional(session.app_name, session.user_id, user_updates)

      for k, v in session_updates.items():
        session.state[k] = v

      batch = self.client.batch()
      batch.update(
          session_ref,
          {
              "state": session.state,
              "updateTime": firestore.SERVER_TIMESTAMP,
          },
      )

      event_id = event.id
      event_ref = session_ref.collection(self.events_collection).document(
          event_id
      )
      event_data = event.model_dump(exclude_none=True, mode="json")
      batch.set(
          event_ref,
          {
              "event_data": event_data,
              "timestamp": firestore.SERVER_TIMESTAMP,
              "appName": session.app_name,
              "userId": session.user_id,
          },
      )

      await batch.commit()
    else:
      batch = self.client.batch()
      event_id = event.id
      event_ref = session_ref.collection(self.events_collection).document(
          event_id
      )
      event_data = event.model_dump(exclude_none=True, mode="json")
      batch.set(
          event_ref,
          {
              "event_data": event_data,
              "timestamp": firestore.SERVER_TIMESTAMP,
              "appName": session.app_name,
              "userId": session.user_id,
          },
      )
      batch.update(session_ref, {"updateTime": firestore.SERVER_TIMESTAMP})
      await batch.commit()

    await super().append_event(session, event)
    return event
