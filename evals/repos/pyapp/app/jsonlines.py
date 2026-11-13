"""JSON Lines parsing."""

import json


def loads_lines(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]
