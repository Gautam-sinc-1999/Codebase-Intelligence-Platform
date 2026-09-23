"""Tracing and evaluation. Optional at runtime — the product works identically without it."""
from app.observability.tracing import (
    tracing_enabled,
    start_trace,
    start_trace_sync,
    span,
    generation,
    set_trace_io,
    score,
    score_trace,
    current_trace_id,
    flush,
    shutdown_tracing,
)

__all__ = [
    "tracing_enabled", "start_trace", "start_trace_sync", "span", "generation",
    "set_trace_io", "score", "score_trace", "current_trace_id", "flush", "shutdown_tracing",
]
