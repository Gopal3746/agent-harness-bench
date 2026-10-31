from collections.abc import Callable

import pytest
from audit_events import (
    build_billing_event,
    build_inventory_event,
    build_user_event,
)

EventBuilder = Callable[
    [str, dict[str, str] | None],
    dict[str, object],
]

EVENT_CASES: tuple[
    tuple[EventBuilder, str, str],
    ...,
] = (
    (
        build_user_event,
        "user.updated",
        "users",
    ),
    (
        build_billing_event,
        "billing.charged",
        "billing",
    ),
    (
        build_inventory_event,
        "inventory.reserved",
        "inventory",
    ),
)


@pytest.mark.parametrize(
    ("builder", "event_name", "_source"),
    EVENT_CASES,
)
def test_builder_sets_event_name(
    builder: EventBuilder,
    event_name: str,
    _source: str,
) -> None:
    event = builder("resource-1", None)

    assert event["event"] == event_name


@pytest.mark.parametrize(
    ("builder", "event_name", "source"),
    EVENT_CASES,
)
def test_builder_sets_identity_and_source(
    builder: EventBuilder,
    event_name: str,
    source: str,
) -> None:
    event = builder("resource-1", None)

    assert event == {
        "event": event_name,
        "resource_id": "resource-1",
        "source": source,
        "metadata": {},
    }


@pytest.mark.parametrize(
    ("builder", "_event_name", "_source"),
    EVENT_CASES,
)
def test_builder_copies_metadata(
    builder: EventBuilder,
    _event_name: str,
    _source: str,
) -> None:
    metadata = {
        "request_id": "request-123",
    }

    event = builder(
        "resource-1",
        metadata,
    )

    event_metadata = event["metadata"]

    assert isinstance(event_metadata, dict)
    assert event_metadata == metadata
    assert event_metadata is not metadata


@pytest.mark.parametrize(
    ("builder", "_event_name", "_source"),
    EVENT_CASES,
)
def test_builder_rejects_blank_resource_id(
    builder: EventBuilder,
    _event_name: str,
    _source: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="resource_id",
    ):
        builder("   ", None)
