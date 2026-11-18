"""OpenAI SDK integration and OpenAI-compatible Responses streaming backend."""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from .backend import ModelRequest
from .events import EventEmitter
from .responses import (
    RAW_ITEMS_KEY,
    ResponseTurn,
    output_text,
    plain_item,
    portable_history,
    public_call,
    raw_state,
    reasoning_text,
    request_input,
    response_status,
    response_tools,
    response_usage,
)
from .types import AgentInterrupted, ModelTurn, ToolCall

RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
FAILOVER_STATUS = frozenset({401, 403, 404})
OVERFLOW_STATUS = frozenset({400, 413, 422})
RETRY_SLEEPS = (0.5, 1.5, 3.0)
STREAM_CLOSE_TIMEOUT = 2.0


async def emit_callback(cb: Callable[..., Any] | None, *args: Any) -> None:
    if cb and asyncio.iscoroutine(maybe := cb(*args)):
        await maybe


class RetryableResponseError(Exception):
    """Failed before any output, so the request can safely be replayed."""

    def __init__(self, cause: BaseException) -> None:
        self.cause = cause
        super().__init__(str(cause))


_RETRY_ERRORS = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
    }
)
_TRANSIENT_ERRORS = _RETRY_ERRORS | {"APIConnectionError", "APITimeoutError"}


def is_retryable_response_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in RETRY_STATUS
    if type(exc).__name__ in _RETRY_ERRORS or any(
        cls.__name__ in _TRANSIENT_ERRORS for cls in type(exc).__mro__
    ):
        return True
    return "overloaded" in f"{exc} {getattr(exc, 'body', '')}".lower()


def can_failover_response_error(exc: BaseException) -> bool:
    """Auth/model errors may fail over; transport errors retry locally first."""

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return False
    return is_retryable_response_error(exc) or getattr(exc, "status_code", None) in FAILOVER_STATUS


def is_context_overflow(exc: BaseException) -> bool:
    if getattr(exc, "status_code", None) not in OVERFLOW_STATUS:
        return False
    text = f"{exc} {getattr(exc, 'body', '')}".lower()
    return any(
        key in text
        for key in (
            "context_length",
            "context length",
            "maximum context",
            "too many tokens",
            "prompt is too long",
            "input is too long",
        )
    )


def is_tool_history_error(exc: BaseException) -> bool:
    """Some compatible Responses backends reject valid raw tool-call replay."""

    if getattr(exc, "status_code", None) != 400:
        return False
    text = f"{exc} {getattr(exc, 'body', '')}".lower()
    return "tool_call" in text and (
        "must be followed by tool messages" in text or "did not have response messages" in text
    )


async def _open_stream(client: AsyncOpenAI, kwargs: dict[str, Any]) -> Any:
    try:
        return await client.responses.create(**kwargs)
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        raise AgentInterrupted() from e


def _consume_close_result(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


async def stream_response_once(
    client: AsyncOpenAI,
    kwargs: dict[str, Any],
    *,
    on_delta,
    on_thinking,
    on_answer_to_thinking,
    on_finish,
    close_timeout: float = STREAM_CLOSE_TIMEOUT,
) -> ResponseTurn:
    """Read one Responses stream into a provider-neutral ``ResponseTurn``."""

    try:
        stream = await _open_stream(client, kwargs)
    except AgentInterrupted:
        raise
    except Exception as e:
        raise RetryableResponseError(e) from e
    parts: list[str] = []
    output: list[dict[str, Any]] = []
    saw_tools = emitted = False
    response = None
    try:
        async for event in stream:
            kind = getattr(event, "type", "")
            if kind == "response.output_item.done":
                item = plain_item(event.item)
                output.append(item)
                if item.get("type") == "function_call":
                    if not saw_tools:
                        saw_tools = True
                        await emit_callback(on_answer_to_thinking)
                elif item.get("type") == "message":
                    text = output_text([item])
                    if text and text != "".join(parts):
                        emitted = True
                        parts.append(text)
                        await emit_callback(on_thinking if saw_tools else on_delta, text)
                elif (
                    item.get("type") == "reasoning"
                    and not emitted
                    and (text := reasoning_text(item))
                ):
                    emitted = True
                    await emit_callback(on_thinking, text)
            elif kind in (
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
            ):
                if text := getattr(event, "delta", ""):
                    emitted = True
                    await emit_callback(on_thinking, text)
            elif kind == "response.output_text.delta":
                if text := getattr(event, "delta", ""):
                    emitted = True
                    parts.append(text)
                    await emit_callback(on_thinking if saw_tools else on_delta, text)
            elif kind in ("response.completed", "response.incomplete"):
                response = event.response
                final_output = [
                    plain_item(item) for item in getattr(response, "output", None) or []
                ]
                if final_output:
                    output = final_output
                    if not parts and (text := output_text(output)):
                        parts.append(text)
                        emitted = True
                        await emit_callback(on_thinking if saw_tools else on_delta, text)
                finish, _ = response_status(response)
                await emit_callback(on_finish, finish)
            elif kind == "response.failed":
                response = event.response
                error = getattr(response, "error", None)
                raise RuntimeError(
                    getattr(error, "message", None) or "Responses API request failed"
                )
            elif kind == "response.error":
                error = getattr(event, "error", None)
                raise RuntimeError(getattr(error, "message", None) or "Responses API stream failed")
    except AgentInterrupted:
        raise
    except (asyncio.CancelledError, KeyboardInterrupt) as e:
        raise AgentInterrupted("".join(parts)) from e
    except Exception as e:
        if emitted:
            raise
        raise RetryableResponseError(e) from e
    finally:
        if close := getattr(stream, "close", None):
            with suppress(Exception):
                result = close()
                if asyncio.iscoroutine(result):
                    interrupted = isinstance(sys.exc_info()[1], AgentInterrupted)
                    closing = asyncio.create_task(result)
                    try:
                        await asyncio.wait_for(asyncio.shield(closing), timeout=close_timeout)
                    except BaseException:
                        closing.cancel()
                        closing.add_done_callback(_consume_close_result)
                        if not interrupted:
                            raise
    finish = response_status(response)[0] if response is not None else None
    return ResponseTurn(
        text="".join(parts),
        output=output,
        usage=response_usage(response),
        finish_reason=finish,
    )


def completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    provider_id: str | None,
    tools: list[dict[str, Any]] | None = None,
    *,
    vision: bool = True,
) -> dict[str, Any]:
    inputs, instructions = request_input(messages, provider_id, vision=vision)
    kwargs: dict[str, Any] = {
        "model": model,
        "input": inputs,
        "tools": response_tools(tools or []),
        "stream": True,
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    if provider_id == "deepseek" and portable_history(messages, provider_id):
        # DeepSeek rejects thinking-mode replay with portable legacy history.
        kwargs["reasoning"] = {"effort": "none"}
    return kwargs


async def stream_response_with_retries(
    client: AsyncOpenAI,
    kwargs: dict[str, Any],
    *,
    on_delta,
    on_thinking,
    on_answer_to_thinking,
    on_finish,
    retry_sleeps: tuple[float, ...] = RETRY_SLEEPS,
    close_timeout: float = STREAM_CLOSE_TIMEOUT,
) -> ResponseTurn:
    """Retry one endpoint without replaying streams that already emitted output."""

    for attempt in range(len(retry_sleeps) + 1):
        try:
            return await stream_response_once(
                client,
                kwargs,
                on_delta=on_delta,
                on_thinking=on_thinking,
                on_answer_to_thinking=on_answer_to_thinking,
                on_finish=on_finish,
                close_timeout=close_timeout,
            )
        except RetryableResponseError as e:
            if attempt == len(retry_sleeps) or not is_retryable_response_error(e.cause):
                raise
            await asyncio.sleep(retry_sleeps[attempt])
    raise AssertionError("unreachable")  # pragma: no cover


async def stream_response_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    on_delta,
    on_thinking,
    on_answer_to_thinking,
    on_finish=None,
    provider_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    vision: bool = True,
    retry_sleeps: tuple[float, ...] = RETRY_SLEEPS,
    close_timeout: float = STREAM_CLOSE_TIMEOUT,
) -> ResponseTurn:
    try:
        return await stream_response_with_retries(
            client,
            completion_kwargs(model, messages, provider_id, tools, vision=vision),
            on_delta=on_delta,
            on_thinking=on_thinking,
            on_answer_to_thinking=on_answer_to_thinking,
            on_finish=on_finish,
            retry_sleeps=retry_sleeps,
            close_timeout=close_timeout,
        )
    except RetryableResponseError as e:
        raise e.cause from None


def _default_vision_support(model: str | None) -> bool:
    name = (model or "").lower()
    hints = (
        "text-",
        "embedding",
        "embed",
        "whisper",
        "tts",
        "rerank",
        "bge-",
        "e5-",
        "moderation",
    )
    return not any(hint in name for hint in hints)


class OpenAIResponsesBackend:
    """One OpenAI-compatible Responses endpoint implementing ModelBackend."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        provider_id: str | None = None,
        context_limit: int | None = None,
        supports_vision: Callable[[str | None], bool] | None = None,
        retry_sleeps: tuple[float, ...] = RETRY_SLEEPS,
        stream_close_timeout: float = STREAM_CLOSE_TIMEOUT,
    ) -> None:
        self.client = client
        self._model = model
        self._provider_id = provider_id
        self._context_limit = context_limit
        self._supports_vision = supports_vision or _default_vision_support
        self.retry_sleeps = retry_sleeps
        self.stream_close_timeout = stream_close_timeout

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider_id(self) -> str | None:
        return self._provider_id

    @property
    def context_limit(self) -> int | None:
        return self._context_limit

    async def close(self) -> None:
        """Close the endpoint client when this backend is owned by an Agent."""

        close = getattr(self.client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    def model_turn(
        self,
        turn: ResponseTurn,
        *,
        provider_id: str | None,
        model: str,
    ) -> ModelTurn:
        calls = []
        for raw in turn.calls:
            call = public_call(raw)
            calls.append(
                ToolCall(
                    id=call["id"],
                    name=call["function"]["name"],
                    arguments=call["function"]["arguments"],
                )
            )
        return ModelTurn(
            text=turn.text,
            calls=calls,
            usage=turn.usage,
            finish_reason=turn.finish_reason,
            model=model,
            state={RAW_ITEMS_KEY: raw_state(provider_id, turn.output)},
        )

    async def complete(self, request: ModelRequest, events: EventEmitter) -> ModelTurn:
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

        turn = await stream_response_completion(
            self.client,
            model=self._model,
            messages=request.messages,
            on_delta=on_delta,
            on_thinking=on_thinking,
            on_answer_to_thinking=on_answer_to_thinking,
            on_finish=on_finish,
            provider_id=self._provider_id,
            tools=request.tools,
            vision=self._supports_vision(self._model),
            retry_sleeps=self.retry_sleeps,
            close_timeout=self.stream_close_timeout,
        )
        return self.model_turn(turn, provider_id=self._provider_id, model=self._model)
