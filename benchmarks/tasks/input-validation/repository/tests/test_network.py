import pytest
from validation_app.network import parse_port


def test_parse_port_accepts_valid_value() -> None:
    assert parse_port("443") == 443


@pytest.mark.parametrize(
    "value",
    ["0", "65536", "not-a-port"],
)
def test_parse_port_rejects_invalid_value(value: str) -> None:
    with pytest.raises(ValueError, match="port"):
        parse_port(value)
