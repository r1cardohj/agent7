"""HTTP header helpers."""


def parse_retry_after(value, now=None):
    if value is None:
        return None
    return float(value)
