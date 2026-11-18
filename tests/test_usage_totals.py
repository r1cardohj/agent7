"""Real token spend: the per-run and per-session totals behind /status and the bar."""

import unittest
from pathlib import Path

from agent7.session import Session, TokenTotals, fmt_rounds, format_usage


def make_session(*, limit: int | None = None, prompt: int | None = None) -> Session:
    session = Session(model="m", cwd=Path("."), client=None)
    session.context_limit = limit
    session.prompt_tokens = prompt
    return session


class TokenTotalsTests(unittest.TestCase):
    def test_add_sums_integers_and_counts_rounds(self) -> None:
        totals = TokenTotals()

        totals.add({"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
        totals.add({"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25})

        self.assertEqual(
            (totals.prompt_tokens, totals.completion_tokens, totals.total_tokens), (30, 7, 37)
        )
        self.assertEqual(totals.rounds, 2)
        self.assertTrue(totals.known)

    def test_unknown_and_zero_are_distinct(self) -> None:
        totals = TokenTotals()
        self.assertIsNone(totals.total)
        self.assertFalse(totals.known)

        totals.add({"total_tokens": 0})

        self.assertEqual(totals.total, 0)
        self.assertTrue(totals.known)

    def test_total_falls_back_to_prompt_plus_completion(self) -> None:
        totals = TokenTotals()

        totals.add({"prompt_tokens": 30, "completion_tokens": 4})

        self.assertEqual(totals.total, 34)

    def test_ignores_bools_missing_and_none_fields(self) -> None:
        totals = TokenTotals()

        totals.add({"prompt_tokens": True, "total_tokens": 5, "completion_tokens": None})

        self.assertIsNone(totals.prompt_tokens)
        self.assertIsNone(totals.completion_tokens)
        self.assertEqual(totals.total, 5)

    def test_none_usage_changes_nothing(self) -> None:
        totals = TokenTotals()

        totals.add(None)
        totals.add({})

        self.assertFalse(totals.known)
        self.assertIsNone(totals.total)

    def test_reset_clears_everything(self) -> None:
        totals = TokenTotals()
        totals.add({"prompt_tokens": 5, "total_tokens": 5})

        totals.reset()

        self.assertIsNone(totals.total)
        self.assertEqual(totals.rounds, 0)


class SessionUsageTests(unittest.TestCase):
    def test_run_total_replaces_and_session_total_accumulates(self) -> None:
        session = make_session()

        session.record_run_usage(
            {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}, rounds=2
        )
        self.assertEqual(session.run_tokens.total, 110)
        self.assertEqual(session.run_tokens.rounds, 2)
        self.assertEqual(session.session_tokens.total, 110)

        session.record_run_usage(
            {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}, rounds=3
        )

        self.assertEqual(session.run_tokens.total, 220)  # this run only
        self.assertEqual(session.session_tokens.total, 330)  # both runs
        self.assertEqual(session.session_tokens.rounds, 5)

    def test_update_usage_stays_the_last_round_for_the_gauge(self) -> None:
        session = make_session(limit=1000)

        session.update_usage({"prompt_tokens": 300, "completion_tokens": 30, "total_tokens": 330})

        self.assertEqual(session.prompt_tokens, 300)
        # Spend is recorded per finished run, never per streamed round.
        self.assertFalse(session.run_tokens.known)

    def test_a_run_without_reported_usage_leaves_totals_unknown(self) -> None:
        session = make_session()

        session.record_run_usage(None, rounds=2)

        self.assertIsNone(session.run_tokens.total)
        self.assertIsNone(session.spent_tokens)

    def test_spent_tokens_prefers_the_session_total(self) -> None:
        session = make_session()

        session.record_run_usage({"total_tokens": 100})
        self.assertEqual(session.spent_tokens, 100)

        session.record_run_usage({"total_tokens": 50})
        self.assertEqual(session.spent_tokens, 150)

    def test_reset_usage_forgets_another_conversation(self) -> None:
        session = make_session(limit=1000)
        session.update_usage({"prompt_tokens": 10})
        session.record_run_usage({"total_tokens": 10})

        session.reset_usage()

        self.assertIsNone(session.prompt_tokens)
        self.assertIsNone(session.spent_tokens)


class FormatUsageTests(unittest.TestCase):
    def test_reports_context_run_and_session(self) -> None:
        session = make_session(limit=262_144, prompt=4_868)
        session.record_run_usage(
            {"prompt_tokens": 9_000, "completion_tokens": 500, "total_tokens": 9_500}, rounds=3
        )

        text = format_usage(session)

        self.assertIn("context=4,868/256k", text)
        self.assertIn("run=9,500 (3 rounds)", text)
        self.assertIn("session=9,500 (prompt=9,000 completion=500)", text)

    def test_no_usage_at_all(self) -> None:
        self.assertEqual(
            format_usage(make_session(limit=128_000)), "usage=none  context_limit=128k"
        )

    def test_known_usage_without_a_context_limit(self) -> None:
        session = make_session(prompt=1_200)
        session.record_run_usage({"total_tokens": 1_400})

        text = format_usage(session)

        self.assertIn("context=1,200", text)
        self.assertIn("run=1,400 (1 round)", text)

    def test_round_wording(self) -> None:
        self.assertEqual(fmt_rounds(1), "1 round")
        self.assertEqual(fmt_rounds(4), "4 rounds")


if __name__ == "__main__":
    unittest.main()
