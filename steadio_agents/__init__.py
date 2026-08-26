"""SteadIO integration for the OpenAI Agents SDK.

Per-run cost and spend-cap visibility for agent runs, as a ``TracingProcessor``.
See README.md for setup.
"""

from .pricing import DEFAULT_PRICING, ModelPricing, cost_cents, format_cents, get_pricing
from .tracing import (
    CAP_OK,
    CAP_OVER,
    CAP_UNKNOWN,
    CAP_WARN,
    ModelUsage,
    RunCost,
    SteadioTracingProcessor,
)

__all__ = [
    "SteadioTracingProcessor",
    "RunCost",
    "ModelUsage",
    "CAP_UNKNOWN",
    "CAP_OK",
    "CAP_WARN",
    "CAP_OVER",
    "DEFAULT_PRICING",
    "ModelPricing",
    "cost_cents",
    "get_pricing",
    "format_cents",
]

__version__ = "0.1.0"
