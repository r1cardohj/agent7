"""agent7's CLI-specific search guidance over the runtime provider."""

import unittest
from unittest.mock import patch

from agent0_runtime import Credential
from agent0_runtime.web_search import _HttpError

from agent7.web_search import Agent7SearchProvider


class Agent7SearchProviderTests(unittest.TestCase):
    def test_missing_credentials_add_the_login_command(self) -> None:
        provider = Agent7SearchProvider(lambda: None)
        with patch("agent0_runtime.web_search._post") as post:
            result = provider.search("latest release")

        post.assert_not_called()
        self.assertIn("credential resolver", result)
        self.assertIn("/login deepseek", result)

    def test_rejected_credentials_suggest_replacing_the_key(self) -> None:
        credential = Credential("sk-old", "https://api.deepseek.com", "stored")
        provider = Agent7SearchProvider(lambda: credential)
        with (
            patch(
                "agent0_runtime.web_search._post",
                side_effect=_HttpError(401, "invalid key"),
            ),
            patch("agent0_runtime.web_search.time.sleep"),
        ):
            result = provider.search("latest release")

        self.assertIn("401", result)
        self.assertIn("/login deepseek", result)


if __name__ == "__main__":
    unittest.main()
