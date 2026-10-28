from collections.abc import Callable


def charge_customer(
    operation: Callable[[], str],
    *,
    max_attempts: int = 3,
) -> str:
    """Charge a customer through the billing service."""

    del max_attempts
    return operation()
