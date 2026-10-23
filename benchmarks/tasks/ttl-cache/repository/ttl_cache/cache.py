from __future__ import annotations

import time
from collections.abc import Callable


class TtlCache[K, V]:
    """Small in-memory cache awaiting expiration support."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._entries: dict[K, V] = {}

    def set(
        self,
        key: K,
        value: V,
        *,
        ttl_seconds: float,
    ) -> None:
        del ttl_seconds
        self._entries[key] = value

    def get(self, key: K) -> V | None:
        return self._entries.get(key)

    def delete(self, key: K) -> bool:
        return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)
