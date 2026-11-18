"""Compatibility exports for the runtime's routed Responses backend."""

from agent0_runtime.routing_backend import RoutedResponsesBackend, stream_response_failover

__all__ = ["RoutedResponsesBackend", "ResponsesBackend", "stream_response_failover"]

ResponsesBackend = RoutedResponsesBackend
