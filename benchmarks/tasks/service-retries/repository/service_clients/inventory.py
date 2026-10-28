from collections.abc import Callable


def reserve_inventory(
    operation: Callable[[], str],
    *,
    max_attempts: int = 3,
) -> str:
    """Reserve an item through the inventory service."""

    del max_attempts
    return operation()
