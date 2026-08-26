"""Token -> integer-cents cost math, mirroring the SteadIO cost engine.

The wire convention is the one thing to get right: SteadIO stores **cents per
one million tokens** (not USD per token, which is what most published pricing
tables use), and a single request's cost is an **integer number of cents**.

This module is a faithful port of ``calculateCostCents`` in
``packages/cost-engine/src/pricing.ts``:

* cost = ceil(in/1e6 * inputCentsPerMillion + out/1e6 * outputCentsPerMillion)
* an unknown model costs 0 -- deliberately, so a new model name never crashes a
  run. Callers that care are told which models were unpriced.

The table below is the cost engine's hand-maintained ``DEFAULT_PRICING_TABLE``
floor, not its full LiteLLM-synced snapshot. It exists so a local run reports
something sensible offline; SteadIO's own ledger remains authoritative.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, NamedTuple, Optional


class ModelPricing(NamedTuple):
    """Rates for one model, in cents per 1,000,000 tokens."""

    input_cents_per_million: float
    output_cents_per_million: float


PricingTable = Mapping[str, ModelPricing]

#: Mirrors DEFAULT_PRICING_TABLE in packages/cost-engine/src/pricing.ts.
DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # Anthropic Claude
    "claude-opus-4-8": ModelPricing(1500, 7500),
    "claude-sonnet-4-6": ModelPricing(300, 1500),
    "claude-haiku-4-5-20251001": ModelPricing(80, 400),
    # OpenAI
    "gpt-4o": ModelPricing(250, 1000),
    "gpt-4o-mini": ModelPricing(15, 60),
    "gpt-4-turbo": ModelPricing(1000, 3000),
    "gpt-3.5-turbo": ModelPricing(50, 150),
    # Gemini
    "gemini-2.0-flash": ModelPricing(10, 40),
    "gemini-1.5-pro": ModelPricing(125, 500),
}


def get_pricing(model: str, table: Optional[PricingTable] = None) -> Optional[ModelPricing]:
    """Look up a model's rates, or None when the model is not in the table."""
    return (table if table is not None else DEFAULT_PRICING).get(model)


def cost_cents(
    model: str,
    input_tokens: int,
    output_tokens: int,
    table: Optional[PricingTable] = None,
) -> int:
    """Integer cents for one request. Unknown model -> 0 (see module docstring)."""
    pricing = get_pricing(model, table)
    if pricing is None:
        return 0
    input_cost = (input_tokens / 1_000_000) * pricing.input_cents_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_cents_per_million
    return math.ceil(input_cost + output_cost)


def format_cents(cents: int) -> str:
    """Cents -> a USD string. Rates and costs are never displayed bare."""
    return f"${cents / 100:.2f}"
