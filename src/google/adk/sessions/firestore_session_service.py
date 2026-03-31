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

from google.cloud import firestore
from pydantic import BaseModel

from ..events.event import Event
from .base_session_service import BaseSessionService
from .base_session_service import GetSessionConfig
from .base_session_service import ListSessionsResponse
from .session import Session

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
    if not session_id:
      from google.adk.platform import uuid as platform_uuid

      session_id = platform_uuid.new_uuid()

    initial_state = state or {}
    now = firestore.SERVER_TIMESTAMP

    session_ref = self._get_sessions_ref(user_id).document(session_id)

    # Check if session already exists
    doc = await session_ref.get()
    if doc.exists:
      from ..errors.already_exists_error import AlreadyExistsError

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
        # The Java code serializes individual fields, but Python schema/v1 uses
        # JSON serialization of the whole event. We'll stick to Pythonic JSON
        # serialization (event.model_dump) for consistency with Python ADK.
        ed = event_data["event_data"]
        # Restore timestamp if needed, or assume it's in event_data
        events.append(Event.model_validate(ed))

    # Let's continue getting session.
    session_state = data.get("state", {})

    # Convert timestamp
    update_time = data.get("updateTime")
    last_update_time = 0.0
    if update_time:
      # If it's a datetime object (Firestore might return it)
      if isinstance(update_time, datetime):
        last_update_time = update_time.timestamp()
      else:
        # Assuming it's a string or float
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
    # If user_id is provided, we can list directly.
    # If not, we might need a collection group query or list all users first.
    # Java listSessions takes appName and userId. It always scopes to user.
    # Python list_sessions has user_id optional.
    # If user_id is None, we should list all sessions for the app across all users.
    # This requires a collection group query on 'sessions'.
    if user_id:
      query = self._get_sessions_ref(user_id).where("appName", "==", app_name)
      docs = await query.get()
    else:
      # Collection group query
      query = self.client.collection_group(self.sessions_collection).where(
          "appName", "==", app_name
      )
      docs = await query.get()

    sessions = []
    for doc in docs:
      data = doc.to_dict()
      if data:
        # Session state is empty for listing as per in_memory
        sessions.append(
            Session(
                id=data["id"],
                app_name=data["appName"],
                user_id=data["userId"],
                state={},  # Empty state for listing
                events=[],  # Empty events for listing
                last_update_time=0.0,  # Or parse from updateTime
            )
        )

    return ListSessionsResponse(sessions=sessions)

  async def delete_session(
      self, *, app_name: str, user_id: str, session_id: str
  ) -> None:
    """Deletes a session and its events from Firestore."""
    session_ref = self._get_sessions_ref(user_id).document(session_id)

    # Delete events subcollection first (Firestore requires manual subcollection deletion)
    events_ref = session_ref.collection(self.events_collection)
    events_docs = await events_ref.get()

    # Batch delete
    batch = self.client.batch()
    for event_doc in events_docs:
      batch.delete(event_doc.reference)
    await batch.commit()

    # Delete session doc
    await session_ref.delete()

  async def append_event(self, session: Session, event: Event) -> Event:
    """Appends an event to a session in Firestore."""
    if event.partial:
      return event

    # Apply temp state to in-memory session (from base class)
    self._apply_temp_state(session, event)
    event = self._trim_temp_delta_state(event)

    session_ref = self._get_sessions_ref(session.user_id).document(session.id)

    # Handle state deltas (app and user state)
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

      # Update session doc with new state and updateTime
      # We'll do it outside the batch or inside if we can.
      # Let's use batch for everything to be atomic.
      # Wait, I didn't add session_ref to batch yet.
      # Let's create a batch.
      batch = self.client.batch()

      if app_updates:
        app_ref = self.client.collection(self.app_state_collection).document(
            session.app_name
        )
        batch.set(app_ref, app_updates, merge=True)

      if user_updates:
        user_ref = (
            self.client.collection(self.user_state_collection)
            .document(session.app_name)
            .collection("users")
            .document(session.user_id)
        )
        batch.set(user_ref, user_updates, merge=True)

      # Update session state in-memory first
      for k, v in session_updates.items():
        session.state[k] = v

      # Update session doc
      batch.update(
          session_ref,
          {
              "state": session.state,
              "updateTime": firestore.SERVER_TIMESTAMP,
          },
      )

      # Add event
      event_id = event.id
      event_ref = session_ref.collection(self.events_collection).document(
          event_id
      )
      # Store event data as JSON serialized string or dict
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
      # No state delta, just add event and update session timestamp
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

    # Also update the in-memory session (adds event to list)
    await super().append_event(session, event)
    return event
