"""In-memory pagination."""


def paginate(items, page, per_page):
    start = (page - 1) * per_page
    end = start + per_page + 1
    return items[start:end]
