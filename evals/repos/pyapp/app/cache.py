"""A tiny memoization decorator."""

from functools import wraps


def memoize(function):
    cache = {}

    @wraps(function)
    def wrapped(*args, **kwargs):
        key = function.__name__
        if key not in cache:
            cache[key] = function(*args, **kwargs)
        return cache[key]

    return wrapped
