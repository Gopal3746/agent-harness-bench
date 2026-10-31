from collections.abc import Mapping


def merge_config(
    base: Mapping[str, object],
    override: Mapping[str, object],
) -> dict[str, object]:
    """Merge an override into a base configuration."""

    return {
        **base,
        **override,
    }
