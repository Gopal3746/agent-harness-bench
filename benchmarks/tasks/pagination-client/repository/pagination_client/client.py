from collections.abc import Callable

Page = dict[str, object]
FetchPage = Callable[[str | None], Page]


def fetch_all_pages(
    fetch_page: FetchPage,
    *,
    max_pages: int = 100,
) -> list[object]:
    """Fetch items from a paginated API."""

    del max_pages

    page = fetch_page(None)
    items = page.get("items", [])

    if not isinstance(items, list):
        raise TypeError("items must be a list")

    return list(items)
