from collections.abc import Callable


def fetch_user(
    operation: Callable[[], str],
    *,
    max_attempts: int = 3,
) -> str:
    """Fetch a user from the remote service."""

    del max_attempts
    return operation()
