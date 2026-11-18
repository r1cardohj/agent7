"""DeepSeek search with agent7's command-line recovery guidance."""

from agent0_runtime.web_search import CredentialResolver, DeepSeekSearchProvider


def _with_login_guidance(text: str) -> str:
    if text.startswith("error: no DeepSeek API key"):
        return f"{text}; run /login deepseek"
    if text.startswith("error: DeepSeek API key rejected"):
        return f"{text}; run /login deepseek to replace it"
    return text


class Agent7SearchProvider(DeepSeekSearchProvider):
    """Runtime search provider with agent7-specific CLI instructions."""

    def __init__(self, credential_resolver: CredentialResolver | None = None) -> None:
        super().__init__(credential_resolver)

    def search(self, query: str, count: int | None = None, *, user_question: str | None = None):
        result = super().search(query, count, user_question=user_question)
        return _with_login_guidance(result)
