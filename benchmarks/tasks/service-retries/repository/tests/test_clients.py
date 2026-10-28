from collections.abc import Callable

import pytest
from service_clients import (
    charge_customer,
    fetch_user,
    reserve_inventory,
)

ClientFunction = Callable[..., str]

CLIENT_CASES: tuple[
    tuple[ClientFunction, str],
    ...,
] = (
    (fetch_user, "user"),
    (charge_customer, "charge"),
    (reserve_inventory, "reservation"),
)


class FlakyOperation:
    def __init__(
        self,
        failures: int,
        result: str,
    ) -> None:
        self.failures = failures
        self.result = result
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1

        if self.calls <= self.failures:
            raise TimeoutError("temporary timeout")

        return self.result


@pytest.mark.parametrize(
    ("client", "expected"),
    CLIENT_CASES,
)
def test_client_returns_immediate_success(
    client: ClientFunction,
    expected: str,
) -> None:
    operation = FlakyOperation(0, expected)

    assert client(operation) == expected
    assert operation.calls == 1


@pytest.mark.parametrize(
    ("client", "expected"),
    CLIENT_CASES,
)
def test_client_retries_timeout_until_success(
    client: ClientFunction,
    expected: str,
) -> None:
    operation = FlakyOperation(2, expected)

    assert client(
        operation,
        max_attempts=3,
    ) == expected
    assert operation.calls == 3


@pytest.mark.parametrize(
    ("client", "expected"),
    CLIENT_CASES,
)
def test_client_stops_after_attempt_limit(
    client: ClientFunction,
    expected: str,
) -> None:
    operation = FlakyOperation(5, expected)

    with pytest.raises(
        TimeoutError,
        match="temporary timeout",
    ):
        client(
            operation,
            max_attempts=3,
        )

    assert operation.calls == 3


@pytest.mark.parametrize(
    ("client", "expected"),
    CLIENT_CASES,
)
def test_client_rejects_invalid_attempt_limit(
    client: ClientFunction,
    expected: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_attempts",
    ):
        client(
            lambda: expected,
            max_attempts=0,
        )


@pytest.mark.parametrize(
    ("client", "_expected"),
    CLIENT_CASES,
)
def test_client_does_not_retry_other_errors(
    client: ClientFunction,
    _expected: str,
) -> None:
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        raise ValueError("invalid request")

    with pytest.raises(
        ValueError,
        match="invalid request",
    ):
        client(
            operation,
            max_attempts=3,
        )

    assert calls == 1
