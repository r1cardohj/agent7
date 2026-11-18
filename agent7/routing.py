"""agent7 provider injection for the runtime-owned routing engine."""

from agent0_runtime import routing as _routing
from agent0_runtime.capabilities import MODEL_CAPABILITIES, ModelCapability
from agent0_runtime.routing import (
    AUTO_REF,
    SCORE_MAX_TOKENS,
    SCORE_TIMEOUT,
    SWITCH_SCORE_MARGIN,
    FailoverRoute,
    RouteSelection,
    RouteTarget,
)

from .providers import list_providers

__all__ = [
    "AUTO_REF",
    "MODEL_CAPABILITIES",
    "SCORE_MAX_TOKENS",
    "SCORE_TIMEOUT",
    "SWITCH_SCORE_MARGIN",
    "FailoverRoute",
    "ModelCapability",
    "RouteSelection",
    "RouteTarget",
    "build_auto_route",
]


def build_auto_route(
    credentials,
    *,
    models=None,
    context_limit=None,
):
    return _routing.build_auto_route(
        credentials,
        models=models,
        context_limit=context_limit,
        provider_specs=list_providers(),
    )
