def build_inventory_event(
    resource_id: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build an audit event for the inventory service."""

    del resource_id, metadata

    return {
        "event": "inventory.reserved",
    }
