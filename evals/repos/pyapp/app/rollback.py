"""Compensating transactions."""


def run_transaction(operations):
    try:
        for do, undo in operations:
            do()
    except Exception:
        for do, undo in operations:
            undo()
        raise
