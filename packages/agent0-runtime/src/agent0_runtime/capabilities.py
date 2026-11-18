"""Reviewed model capabilities shared by routing, image tools, and hosts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCapability:
    """Reviewed model metadata; unknown listed models stay out of auto routing."""

    provider_id: str
    model: str
    context_limit: int
    strengths: frozenset[str]
    scorer_priority: int | None = None
    supports_responses: bool = True
    supports_tools: bool = True
    supports_vision: bool = True
    routing_hint: str = ""

    @property
    def ref(self) -> str:
        return f"{self.provider_id}/{self.model}"


MODEL_CAPABILITIES = (
    ModelCapability(
        provider_id="deepseek",
        model="deepseek-flash",
        context_limit=1_048_576,
        strengths=frozenset({"fast", "coding"}),
        scorer_priority=1,
        supports_vision=True,
        routing_hint="Fast general coding, explanations, and quick edits.",
    ),
    ModelCapability(
        provider_id="kimi",
        model="kimi-for-coding",
        context_limit=1_048_576,
        strengths=frozenset({"fast", "coding"}),
        scorer_priority=2,
        routing_hint="Fast implementation, refactoring, and test-focused coding.",
    ),
    ModelCapability(
        provider_id="kimi",
        model="k3.1-pro",
        context_limit=1_048_576,
        strengths=frozenset({"strong"}),
        routing_hint="Strong architecture and repository-wide coding; largest context.",
    ),
)
