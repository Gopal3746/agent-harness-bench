import pytest
from pagination_client import (
    Page,
    fetch_all_pages,
)


class FakeApi:
    def __init__(
        self,
        pages: dict[str | None, Page],
    ) -> None:
        self.pages = pages
        self.cursors: list[str | None] = []

    def __call__(
        self,
        cursor: str | None,
    ) -> Page:
        self.cursors.append(cursor)
        return self.pages[cursor]


def test_fetches_single_page() -> None:
    api = FakeApi(
        {
            None: {
                "items": [1, 2],
                "next_cursor": None,
            }
        }
    )

    assert fetch_all_pages(api) == [1, 2]
    assert api.cursors == [None]


def test_follows_every_page() -> None:
    api = FakeApi(
        {
            None: {
                "items": [1, 2],
                "next_cursor": "page-2",
            },
            "page-2": {
                "items": [3],
                "next_cursor": "page-3",
            },
            "page-3": {
                "items": [4, 5],
                "next_cursor": None,
            },
        }
    )

    assert fetch_all_pages(api) == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_passes_cursor_to_next_request() -> None:
    api = FakeApi(
        {
            None: {
                "items": [1],
                "next_cursor": "next",
            },
            "next": {
                "items": [2],
                "next_cursor": None,
            },
        }
    )

    fetch_all_pages(api)

    assert api.cursors == [None, "next"]


def test_rejects_invalid_page_limit() -> None:
    api = FakeApi(
        {
            None: {
                "items": [],
                "next_cursor": None,
            }
        }
    )

    with pytest.raises(
        ValueError,
        match="max_pages",
    ):
        fetch_all_pages(
            api,
            max_pages=0,
        )


def test_raises_when_page_limit_is_reached() -> None:
    api = FakeApi(
        {
            None: {
                "items": [1],
                "next_cursor": "page-2",
            },
            "page-2": {
                "items": [2],
                "next_cursor": "page-3",
            },
            "page-3": {
                "items": [3],
                "next_cursor": None,
            },
        }
    )

    with pytest.raises(
        RuntimeError,
        match="max_pages",
    ):
        fetch_all_pages(
            api,
            max_pages=2,
        )


def test_rejects_repeated_cursor() -> None:
    api = FakeApi(
        {
            None: {
                "items": [1],
                "next_cursor": "loop",
            },
            "loop": {
                "items": [2],
                "next_cursor": "loop",
            },
        }
    )

    with pytest.raises(
        ValueError,
        match="cursor",
    ):
        fetch_all_pages(api)
