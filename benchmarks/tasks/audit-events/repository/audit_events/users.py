def build_user_event(
    resource_id: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build an audit event for the user service."""

    del resource_id, metadata

    return {
        "event": "user.updated",
    }
