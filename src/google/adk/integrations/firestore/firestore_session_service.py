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
from typing import cast
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from google.cloud import firestore

from pydantic import BaseModel

from ...events.event import Event
from ...sessions import _session_util
from ...sessions.base_session_service import BaseSessionService
from ...sessions.base_session_service import GetSessionConfig
from ...sessions.base_session_service import ListSessionsResponse
from ...sessions.session import Session
from ...sessions.state import State

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_ROOT_COLLECTION = "adk-session"
DEFAULT_SESSIONS_COLLECTION = "sessions"
DEFAULT_EVENTS_COLLECTION = "events"
DEFAULT_APP_STATE_COLLECTION = "app_states"
DEFAULT_USER_STATE_COLLECTION = "user_states"


class FirestoreSessionService(BaseSessionService):  # type: ignore[misc]
  """Session service that uses Google Cloud Firestore as the backend.

  Hierarchy for sessions:
  adk-session
  ↳ <app name>
    ↳ users
      ↳ <user ID>
        ↳ sessions
          ↳ <session ID>
            ↳ events
              ↳ <event ID>

  Hierarchy for shared App/User state configurations:
  app_states
  ↳ <app name>

  user_states
  ↳ <app name>
    ↳ users
      ↳ <user ID>
  """

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
      merged_state[State.APP_PREFIX + key] = value
    for key, value in user_state.items():
      merged_state[State.USER_PREFIX + key] = value
    return merged_state

  def _get_sessions_ref(
      self, app_name: str, user_id: str
  ) -> firestore.AsyncCollectionReference:
    return (
        self.client.collection(self.root_collection)
        .document(app_name)
        .collection("users")
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

    session_ref = self._get_sessions_ref(app_name, user_id).document(session_id)

    # Extract state deltas
    state_deltas = _session_util.extract_state_delta(initial_state)
    app_state_delta = state_deltas["app"]
    user_state_delta = state_deltas["user"]
    session_state = state_deltas["session"]

    app_ref = self.client.collection(self.app_state_collection).document(
        app_name
    )
    user_ref = (
        self.client.collection(self.user_state_collection)
        .document(app_name)
        .collection("users")
        .document(user_id)
    )

    session_data = {
        "id": session_id,
        "appName": app_name,
        "userId": user_id,
        "state": session_state,
        "createTime": now,
        "updateTime": now,
    }

    @firestore.async_transactional  # type: ignore[untyped-decorator]
    async def _create_txn(transaction: firestore.AsyncTransaction) -> None:
      # 1. Reads
      snap = await session_ref.get(transaction=transaction)
      if snap.exists:
        from ...errors.already_exists_error import AlreadyExistsError

        raise AlreadyExistsError(f"Session {session_id} already exists.")

      app_snap = (
          await app_ref.get(transaction=transaction)
          if app_state_delta
          else None
      )
      user_snap = (
          await user_ref.get(transaction=transaction)
          if user_state_delta
          else None
      )

      # 2. Writes
      if app_state_delta:
        current_app = (
            app_snap.to_dict() if (app_snap and app_snap.exists) else {}
        )
        current_app.update(app_state_delta)
        transaction.set(app_ref, current_app, merge=True)

      if user_state_delta:
        current_user = (
            user_snap.to_dict() if (user_snap and user_snap.exists) else {}
        )
        current_user.update(user_state_delta)
        transaction.set(user_ref, current_user, merge=True)

      transaction.set(session_ref, session_data)

    transaction_obj = self.client.transaction()
    await _create_txn(transaction_obj)

    storage_app_doc = await app_ref.get()
    storage_app_state = (
        storage_app_doc.to_dict() if storage_app_doc.exists else {}
    )
    storage_user_doc = await user_ref.get()
    storage_user_state = (
        storage_user_doc.to_dict() if storage_user_doc.exists else {}
    )

    merged_state = self._merge_state(
        storage_app_state, storage_user_state, session_state
    )

    local_now = datetime.now(timezone.utc).timestamp()

    return Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state=merged_state,
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
    session_ref = self._get_sessions_ref(app_name, user_id).document(session_id)
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

    # Fetch shared state
    app_ref = self.client.collection(self.app_state_collection).document(
        app_name
    )
    user_ref = (
        self.client.collection(self.user_state_collection)
        .document(app_name)
        .collection("users")
        .document(user_id)
    )
    app_doc = await app_ref.get()
    app_state = app_doc.to_dict() if app_doc.exists else {}
    user_doc = await user_ref.get()
    user_state = user_doc.to_dict() if user_doc.exists else {}

    merged_state = self._merge_state(app_state, user_state, session_state)

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
        state=merged_state,
        events=events,
        last_update_time=last_update_time,
    )

  async def list_sessions(
      self, *, app_name: str, user_id: Optional[str] = None
  ) -> ListSessionsResponse:
    """Lists sessions from Firestore."""
    if user_id:
      query = self._get_sessions_ref(app_name, user_id).where(
          "appName", "==", app_name
      )
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
    session_ref = self._get_sessions_ref(app_name, user_id).document(session_id)

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

  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event to a session in Firestore."""
    from google.cloud import firestore

    if event.partial:
      return event

    self._apply_temp_state(session, event)
    event = self._trim_temp_delta_state(event)

    session_ref = self._get_sessions_ref(
        session.app_name, session.user_id
    ).document(session.id)

    if event.actions and event.actions.state_delta:
      state_delta = event.actions.state_delta
      state_deltas = _session_util.extract_state_delta(state_delta)
      app_updates = state_deltas["app"]
      user_updates = state_deltas["user"]
      session_updates = state_deltas["session"]

      app_ref = self.client.collection(self.app_state_collection).document(
          session.app_name
      )
      user_ref = (
          self.client.collection(self.user_state_collection)
          .document(session.app_name)
          .collection("users")
          .document(session.user_id)
      )

      @firestore.async_transactional  # type: ignore[untyped-decorator]
      async def _append_txn(transaction: firestore.AsyncTransaction) -> None:
        # 1. Reads
        app_snap = (
            await app_ref.get(transaction=transaction) if app_updates else None
        )
        user_snap = (
            await user_ref.get(transaction=transaction)
            if user_updates
            else None
        )

        # 2. Writes
        if app_updates and app_snap is not None:
          current_app = app_snap.to_dict() if app_snap.exists else {}
          current_app.update(app_updates)
          transaction.set(app_ref, current_app, merge=True)

        if user_updates and user_snap is not None:
          current_user = user_snap.to_dict() if user_snap.exists else {}
          current_user.update(user_updates)
          transaction.set(user_ref, current_user, merge=True)

        for k, v in session_updates.items():
          session.state[k] = v

        transaction.update(
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
        transaction.set(
            event_ref,
            {
                "event_data": event_data,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "appName": session.app_name,
                "userId": session.user_id,
            },
        )

      transaction_obj = self.client.transaction()
      await _append_txn(transaction_obj)
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
