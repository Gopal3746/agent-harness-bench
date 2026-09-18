import pytest
from validation_app.tags import normalize_tags


def test_normalize_tags_canonicalizes_values() -> None:
    assert normalize_tags([" Python ", "AI"]) == ["python", "ai"]


@pytest.mark.parametrize(
    "tags",
    [[], ["python", "   "]],
)
def test_normalize_tags_rejects_invalid_values(
    tags: list[str],
) -> None:
    with pytest.raises(ValueError, match="tags"):
        normalize_tags(tags)
