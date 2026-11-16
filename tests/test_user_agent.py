import unittest

from agent7 import __version__
from agent7.providers import Agent7Client, make_client


class UserAgentTests(unittest.TestCase):
    def test_client_identifies_as_agent7_not_sdk_default(self) -> None:
        client = make_client("missing", "https://example.invalid/v1")
        self.assertIsInstance(client, Agent7Client)
        self.assertEqual(client.user_agent, f"agent7/{__version__}")
        self.assertNotIn("Python", client.user_agent)
        self.assertNotIn("AsyncOpenAI", client.user_agent)

    def test_default_headers_use_the_agent7_user_agent(self) -> None:
        client = make_client("missing", "https://example.invalid/v1")
        self.assertEqual(client.default_headers["User-Agent"], f"agent7/{__version__}")


if __name__ == "__main__":
    unittest.main()
