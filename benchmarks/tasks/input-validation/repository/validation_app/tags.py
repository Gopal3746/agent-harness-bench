def normalize_tags(tags: list[str]) -> list[str]:
    """Return canonical tags while preserving their order."""

    return [tag.strip().lower() for tag in tags]
