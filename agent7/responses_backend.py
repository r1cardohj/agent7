"""agent7-specific route selection/failover over the core Responses backend."""

from __future__ import annotations

from typing import Any

from .core.events import EventEmitter
from .core.openai_backend import (
    OpenAIResponsesBackend,
    RetryableResponseError,
    can_failover_response_error,
    completion_kwargs,
    emit_callback,
    stream_response_with_retries,
)
from .core.responses import ResponseTurn
from .core.types import ModelTurn
from .images import supports_vision
from .routing import FailoverRoute, RouteTarget


async def stream_response_failover(
    route: FailoverRoute,
    *,
    messages: list[dict[str, Any]],
    on_delta,
    on_thinking,
    on_answer_to_thinking,
    on_finish,
    on_route,
    tools=None,
    supports_model_vision=supports_vision,
    retry_sleeps=(0.5, 1.5, 3.0),
    close_timeout=2.0,
) -> tuple[ResponseTurn, RouteTarget]:
    """Replay only silent failed requests; never duplicate partial output."""

    while True:
        target = route.current
        try:
            turn = await stream_response_with_retries(
                target.client,
                completion_kwargs(
                    target.model,
                    messages,
                    target.provider_id,
                    tools,
                    vision=supports_model_vision(target.model),
                ),
                on_delta=on_delta,
                on_thinking=on_thinking,
                on_answer_to_thinking=on_answer_to_thinking,
                on_finish=on_finish,
                retry_sleeps=retry_sleeps,
                close_timeout=close_timeout,
            )
        except RetryableResponseError as error:
            next_target = route.advance() if can_failover_response_error(error.cause) else None
            if next_target is None:
                raise error.cause from None
            await emit_callback(on_route, next_target, target, error.cause)
        else:
            return turn, target


class RoutedResponsesBackend(OpenAIResponsesBackend):
    """Optional agent7 route selection over a generic single-endpoint backend."""

    def __init__(
        self,
        client,
        model: str,
        *,
        route: FailoverRoute,
        provider_id: str | None = None,
        context_limit: int | None = None,
        on_route=None,
        retry_sleeps=(0.5, 1.5, 3.0),
        stream_close_timeout=2.0,
    ) -> None:
        super().__init__(
            client,
            model,
            supports_vision=supports_vision,
            retry_sleeps=retry_sleeps,
            stream_close_timeout=stream_close_timeout,
        )
        self.route = route
        self._provider_id = provider_id
        self._context_limit = context_limit
        self.on_route = on_route

    @property
    def model(self) -> str:
        return self.route.current.model

    @property
    def provider_id(self) -> str | None:
        return self.route.current.provider_id

    @property
    def context_limit(self) -> int | None:
        return self.route.current.context_limit

    async def _route_changed(
        self,
        events: EventEmitter,
        target: RouteTarget,
        previous: RouteTarget | None,
        error: BaseException | None,
    ) -> None:
        await events.emit(
            "route_changed",
            target=target.ref,
            previous=previous.ref if previous is not None else None,
            error=str(error) if error is not None else None,
            error_type=type(error).__name__ if error is not None else None,
        )
        await emit_callback(self.on_route, target, previous, error)

    async def prepare(self, task: str, messages, estimated_tokens: int, events) -> None:
        await self.route.start(task, messages, estimated_tokens)
        await self._route_changed(events, self.route.current, None, None)

    async def complete(self, request, events: EventEmitter) -> ModelTurn:
        if request.cancellation is not None:
            request.cancellation.throw_if_cancelled(run_id=request.run_id)

        async def on_delta(text) -> None:
            await events.emit("text_delta", turn=request.turn, text=text)

        async def on_thinking(text) -> None:
            await events.emit("reasoning_delta", turn=request.turn, text=text)

        async def on_answer_to_thinking() -> None:
            await events.emit("answer_to_thinking", turn=request.turn)

        async def on_finish(reason) -> None:
            await events.emit("model_finished", turn=request.turn, finish_reason=reason)

        async def changed(current, previous, error) -> None:
            await self._route_changed(events, current, previous, error)

        turn, target = await stream_response_failover(
            self.route,
            messages=request.messages,
            on_delta=on_delta,
            on_thinking=on_thinking,
            on_answer_to_thinking=on_answer_to_thinking,
            on_finish=on_finish,
            on_route=changed,
            tools=request.tools,
            supports_model_vision=self._supports_vision,
            retry_sleeps=self.retry_sleeps,
            close_timeout=self.stream_close_timeout,
        )
        return self.model_turn(turn, provider_id=target.provider_id, model=target.model)

    async def close(self) -> None:
        await self.route.close()


# Compatibility name from the first runtime extraction pass.
ResponsesBackend = RoutedResponsesBackend
