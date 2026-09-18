import pytest
from validation_app.users import normalize_username


def test_normalize_username_canonicalizes_value() -> None:
    assert normalize_username("  Ada.Lovelace  ") == "ada.lovelace"


def test_normalize_username_rejects_blank_value() -> None:
    with pytest.raises(ValueError, match="username"):
        normalize_username("   ")
