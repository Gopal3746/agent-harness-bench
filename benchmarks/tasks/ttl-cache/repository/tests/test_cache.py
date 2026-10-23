from __future__ import annotations

import pytest
from ttl_cache import TtlCache


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_cache_returns_stored_value() -> None:
    cache: TtlCache[str, int] = TtlCache()

    cache.set("answer", 42, ttl_seconds=10)

    assert cache.get("answer") == 42


def test_cache_returns_none_for_missing_key() -> None:
    cache: TtlCache[str, int] = TtlCache()

    assert cache.get("missing") is None


def test_delete_reports_whether_key_existed() -> None:
    cache: TtlCache[str, int] = TtlCache()
    cache.set("answer", 42, ttl_seconds=10)

    assert cache.delete("answer") is True
    assert cache.delete("answer") is False


def test_clear_removes_all_entries() -> None:
    cache: TtlCache[str, int] = TtlCache()
    cache.set("one", 1, ttl_seconds=10)
    cache.set("two", 2, ttl_seconds=10)

    cache.clear()

    assert cache.size == 0


def test_entry_expires_at_deadline() -> None:
    clock = FakeClock()
    cache: TtlCache[str, int] = TtlCache(clock=clock)
    cache.set("answer", 42, ttl_seconds=5)

    clock.advance(5)

    assert cache.get("answer") is None


def test_expired_entry_is_removed_from_size() -> None:
    clock = FakeClock()
    cache: TtlCache[str, int] = TtlCache(clock=clock)
    cache.set("short", 1, ttl_seconds=2)
    cache.set("long", 2, ttl_seconds=10)

    clock.advance(3)

    assert cache.size == 1
    assert cache.get("long") == 2


def test_overwriting_entry_resets_expiration() -> None:
    clock = FakeClock()
    cache: TtlCache[str, int] = TtlCache(clock=clock)
    cache.set("answer", 41, ttl_seconds=5)

    clock.advance(4)
    cache.set("answer", 42, ttl_seconds=5)
    clock.advance(2)

    assert cache.get("answer") == 42

    clock.advance(3)

    assert cache.get("answer") is None


@pytest.mark.parametrize("ttl_seconds", [0, -1])
def test_cache_rejects_non_positive_ttl(
    ttl_seconds: float,
) -> None:
    cache: TtlCache[str, int] = TtlCache()

    with pytest.raises(ValueError, match="ttl_seconds"):
        cache.set(
            "answer",
            42,
            ttl_seconds=ttl_seconds,
        )
