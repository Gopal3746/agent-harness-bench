def build_billing_event(
    resource_id: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build an audit event for the billing service."""

    del resource_id, metadata

    return {
        "event": "billing.charged",
    }
