"""Zero-config intent routing over agent7's confirmed model capability table."""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import AsyncOpenAI

from .providers import Credential, list_providers
from .responses import output_text, plain_item

AUTO_REF = "auto"
SCORE_TIMEOUT = 5.0
SCORE_MAX_TOKENS = 500
SWITCH_SCORE_MARGIN = 3.0
_INTENTS = frozenset({"quick", "coding", "debugging", "reasoning"})

@dataclass(frozen=True)
class ModelCapability:
    """Reviewed auto-routing metadata; unknown listed models stay out of auto."""

    provider_id: str
    model: str
    context_limit: int
    strengths: frozenset[str]
    scorer_priority: int | None = None
    supports_responses: bool = True
    supports_tools: bool = True
    routing_hint: str = ""

    @property
    def ref(self) -> str:
        return f"{self.provider_id}/{self.model}"

MODEL_CAPABILITIES = (
    ModelCapability(
        provider_id="deepseek", model="deepseek-flash", context_limit=1_048_576,
        strengths=frozenset({"fast", "coding"}), scorer_priority=1,
        routing_hint="Fast general coding, explanations, and quick edits.",
    ),
    ModelCapability(
        provider_id="kimi", model="kimi-for-coding", context_limit=1_048_576,
        strengths=frozenset({"fast", "coding"}), scorer_priority=2,
        routing_hint="Fast implementation, refactoring, and test-focused coding.",
    ),
    ModelCapability(
        provider_id="xai", model="grok-build-0.1", context_limit=256_000,
        strengths=frozenset({"fast", "coding"}), scorer_priority=3,
        routing_hint="Fast coding for builds, shell work, and tool-heavy tasks.",
    ),
    ModelCapability(
        provider_id="xai", model="grok-4.7", context_limit=500_000,
        strengths=frozenset({"strong"}),
        routing_hint="Strong general reasoning and root-cause debugging.",
    ),
    ModelCapability(
        provider_id="kimi", model="k3.1-pro", context_limit=1_048_576,
        strengths=frozenset({"strong"}),
        routing_hint="Strong architecture and repository-wide coding; largest context.",
    ),
)
@dataclass(frozen=True)
class RouteTarget:
    """One concrete upstream request destination."""

    provider_id: str
    model: str
    credential: Credential
    client: AsyncOpenAI
    context_limit: int | None = None
    capability: ModelCapability | None = None

    @property
    def ref(self) -> str:
        return f"{self.provider_id}/{self.model}"

@dataclass(frozen=True)
class RouteSelection:
    intent: str
    scores: dict[str, float]
    scorer_ref: str
    error: str = ""

@dataclass
class FailoverRoute:
    """Select once per user task, then keep that order through tool rounds."""

    targets: tuple[RouteTarget, ...]
    intent_routing: bool = False
    index: int = 0
    selection: RouteSelection | None = field(init=False, default=None)
    _active: tuple[RouteTarget, ...] = field(init=False, repr=False)
    _started: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("auto: a route needs at least one target")
        self._active = self.targets

    @property
    def current(self) -> RouteTarget:
        return self._active[self.index]

    @property
    def active_targets(self) -> tuple[RouteTarget, ...]:
        return self._active[self.index:]

    def advance(self) -> RouteTarget | None:
        """Move to the next target, or return None after the last one."""
        if self.index + 1 >= len(self._active):
            return None
        self.index += 1
        return self.current

    async def start(
        self, task: str, messages: list[dict[str, Any]], estimated_tokens: int,
    ) -> None:
        """Choose a new task's order; execution failover never invokes the scorer."""
        # A new route has no incumbent: its first task must use the argmax.
        previous = self.current if self._started else None
        self._started = True
        self.index = 0
        self.selection = None
        self._active = self.targets
        if not self.intent_routing:
            return
        eligible = _context_eligible(self.targets, estimated_tokens)
        self._active = eligible
        if len(eligible) == 1:
            return
        scorer = _scorer_for(eligible, previous)
        try:
            intent, scores = await _score_targets(scorer, eligible, task, messages)
        except Exception as e:  # scoring is an optimization, never a task SPOF
            self._active = _fallback_order(eligible, previous)
            self.selection = RouteSelection("default", {}, scorer.ref, type(e).__name__)
            return
        self._active = _scored_order(eligible, previous, scores)
        self.selection = RouteSelection(intent, scores, scorer.ref)

    async def close(self) -> None:
        seen: set[int] = set()
        for target in self.targets:
            client: Any = target.client
            if id(client) in seen:
                continue
            seen.add(id(client))
            with suppress(Exception):
                result = client.close()
                if asyncio.iscoroutine(result):
                    await result

def build_auto_route(
    credentials: dict[str, Credential], *, models: list[str] | None = None,
    context_limit: Callable[[str], int | None] | None = None,
) -> FailoverRoute:
    """Build auto from reviewed models available in the current login session."""
    providers = {p.id: p for p in list_providers()}
    listed = {ref.lower() for ref in models or []}
    targets: list[RouteTarget] = []
    for capability in MODEL_CAPABILITIES:
        if not capability.supports_responses or not capability.supports_tools:
            continue
        if listed and capability.ref.lower() not in listed:
            continue
        provider = providers.get(capability.provider_id)
        if provider is None:
            continue
        credential = credentials.get(provider.id) or provider.resolve()
        if credential is None:
            continue
        targets.append(RouteTarget(
            provider_id=provider.id,
            model=capability.model,
            credential=credential,
            client=provider.make_client(credential),
            context_limit=(
                context_limit(capability.model)
                if context_limit else capability.context_limit
            ),
            capability=capability,
        ))
    if not targets:
        raise ValueError("auto: no supported models with credentials")
    return FailoverRoute(tuple(targets), intent_routing=True)

def _context_eligible(
    targets: tuple[RouteTarget, ...], estimated_tokens: int,
) -> tuple[RouteTarget, ...]:
    fitting = tuple(
        target for target in targets
        if target.context_limit is None or estimated_tokens < target.context_limit
    )
    if fitting:
        return fitting
    # Nothing fits as-is: pick the largest window and let normal compaction run.
    return (max(targets, key=lambda target: target.context_limit or 0),)

def _scorer_for(
    targets: tuple[RouteTarget, ...], previous: RouteTarget | None,
) -> RouteTarget:
    scorers = [target for target in targets if target.capability and target.capability.scorer_priority]
    if scorers:
        return min(scorers, key=lambda target: target.capability.scorer_priority or 99)
    return previous if previous in targets else targets[0]

def _fallback_order(
    targets: tuple[RouteTarget, ...], previous: RouteTarget | None,
) -> tuple[RouteTarget, ...]:
    if previous not in targets:
        return targets
    return (previous, *(target for target in targets if target is not previous))

def _scored_order(
    targets: tuple[RouteTarget, ...], previous: RouteTarget | None, scores: dict[str, float],
) -> tuple[RouteTarget, ...]:
    ordered = sorted(targets, key=lambda target: -scores.get(target.ref, -1.0))
    if previous in targets and previous.ref in scores:
        best_score = scores.get(ordered[0].ref, -1.0)
        if best_score - scores[previous.ref] < SWITCH_SCORE_MARGIN:
            ordered = [previous, *(target for target in ordered if target is not previous)]
    return tuple(ordered)

def _recent_context(messages: list[dict[str, Any]]) -> str:
    entries: list[str] = []
    for message in messages[:-1]:
        role = message.get("role")
        content = message.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            entries.append(f"{role}: {content.strip()[:2000]}")
    return "\n".join(entries[-2:])[:4000]

async def _score_targets(
    scorer: RouteTarget, targets: tuple[RouteTarget, ...], task: str,
    messages: list[dict[str, Any]],
) -> tuple[str, dict[str, float]]:
    candidates = [{
        "ref": target.ref,
        "strengths": sorted(target.capability.strengths) if target.capability else [],
        "routing_hint": target.capability.routing_hint if target.capability else "",
        "context_limit": target.context_limit,
    } for target in targets]
    payload = {
        "task": task[:6000],
        "recent_context": _recent_context(messages),
        "candidates": candidates,
    }
    instructions = (
        "You are a model router, not the task solver. Score every supplied candidate "
        "from 0 to 100 for this task. Use fast/coding models for simple or quick coding "
        "tasks and strong models for difficult debugging, architecture, and reasoning. "
        "Classify quick as explanations or trivial edits, coding as implementation or tests, "
        "debugging as failures or root causes, and reasoning as design, architecture, "
        "tradeoffs, or proofs. Candidate order is not a ranking; use routing_hint to "
        "differentiate models within a tier. Consider tool use and context needs. "
        "Return only one JSON object with this shape: "
        '{"intent":"quick|coding|debugging|reasoning","scores":{"provider/model":0}}. '
        "Use only the exact candidate refs; do not answer the task."
    )
    kwargs: dict[str, Any] = {
        "model": scorer.model,
        "instructions": instructions,
        "input": json.dumps(payload, ensure_ascii=False),
        "stream": False,
        "temperature": 0,
        "max_output_tokens": SCORE_MAX_TOKENS,
    }
    if scorer.provider_id == "deepseek":
        kwargs["reasoning"] = {"effort": "none"}
    response = await asyncio.wait_for(
        scorer.client.responses.create(**kwargs), timeout=SCORE_TIMEOUT
    )
    output = response.get("output") if isinstance(response, dict) else getattr(response, "output", [])
    text = output_text([plain_item(item) for item in output or []])
    return _parse_scores(text, targets)

def _parse_scores(text: str, targets: tuple[RouteTarget, ...]) -> tuple[str, dict[str, float]]:
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```") and "\n" in raw:
        raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("scores"), dict):
        raise ValueError("auto: scorer returned no scores")
    intent = data.get("intent")
    if intent not in _INTENTS:
        intent = "coding"
    refs = {target.ref.lower(): target.ref for target in targets}
    scores: dict[str, float] = {}
    for ref, value in data["scores"].items():
        canonical = refs.get(str(ref).lower())
        if canonical is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        score = float(value)
        if math.isfinite(score) and 0 <= score <= 100:
            scores[canonical] = score
    if not scores:
        raise ValueError("auto: scorer returned no usable scores")
    return intent, scores
