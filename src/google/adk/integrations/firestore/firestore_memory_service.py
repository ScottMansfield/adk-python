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

import asyncio
import logging
import os
import re
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.cloud.firestore_v1.base_query import FieldFilter
from typing_extensions import override

from ...events.event import Event
from ...memory import _utils
from ...memory.base_memory_service import BaseMemoryService
from ...memory.base_memory_service import SearchMemoryResponse
from ...memory.memory_entry import MemoryEntry

if TYPE_CHECKING:
  from google.cloud import firestore

  from ...sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

DEFAULT_EVENTS_COLLECTION = "events"

DEFAULT_STOP_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "should",
    "shall",
    "may",
    "might",
    "must",
    "up",
    "down",
    "out",
    "in",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
}


class FirestoreMemoryService(BaseMemoryService):
  """Memory service that uses Google Cloud Firestore as the backend."""

  def __init__(
      self,
      client: Optional[firestore.AsyncClient] = None,
      events_collection: Optional[str] = None,
      stop_words: Optional[set[str]] = None,
  ):
    """Initializes the Firestore memory service.

    Args:
      client: An optional Firestore AsyncClient. If not provided, a new one
        will be created.
      events_collection: The name of the events collection or collection group.
        Defaults to 'events'.
      stop_words: A set of words to ignore when extracting keywords. Defaults to
        a standard English stop words list.
    """
    if client is None:
      from google.cloud import firestore

      self.client = firestore.AsyncClient()
    else:
      self.client = client
    self.events_collection = events_collection or DEFAULT_EVENTS_COLLECTION
    self.stop_words = (
        stop_words if stop_words is not None else DEFAULT_STOP_WORDS
    )

  @override
  async def add_session_to_memory(self, session: Session) -> None:
    """No-op. Assumes events are written to Firestore by FirestoreSessionService."""
    pass

  def _extract_keywords(self, text: str) -> set[str]:
    """Extracts keywords from text, ignoring stop words."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    return {word for word in words if word not in self.stop_words}

  async def _search_by_keyword(
      self, app_name: str, user_id: str, keyword: str
  ) -> list[MemoryEntry]:
    """Searches for events matching a single keyword."""
    query = (
        self.client.collection_group(self.events_collection)
        .where(filter=FieldFilter("appName", "==", app_name))
        .where(filter=FieldFilter("userId", "==", user_id))
        .where(filter=FieldFilter("keywords", "array_contains", keyword))
    )

    docs = await query.get()
    entries = []
    for doc in docs:
      data = doc.to_dict()
      if data and "event_data" in data:
        try:
          event = Event.model_validate(data["event_data"])
          if event.content:
            entries.append(
                MemoryEntry(
                    content=event.content,
                    author=event.author,
                    timestamp=_utils.format_timestamp(event.timestamp),
                )
            )
        except Exception as e:
          logger.warning("Failed to parse event from Firestore: %s", e)

    return entries

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Searches memory for events matching the query."""
    keywords = self._extract_keywords(query)
    if not keywords:
      return SearchMemoryResponse()

    tasks = [
        self._search_by_keyword(app_name, user_id, keyword)
        for keyword in keywords
    ]
    results = await asyncio.gather(*tasks)

    seen = set()
    memories = []
    for result_list in results:
      for entry in result_list:
        content_text = ""
        if entry.content and entry.content.parts:
          content_text = " ".join(
              [part.text for part in entry.content.parts if part.text]
          )
        key = (entry.author, content_text, entry.timestamp)
        if key not in seen:
          seen.add(key)
          memories.append(entry)

    return SearchMemoryResponse(memories=memories)
