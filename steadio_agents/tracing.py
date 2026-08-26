"""A SteadIO ``TracingProcessor`` for the OpenAI Agents SDK.

Register it and every agent run reports its token usage, its cost in cents, and
where that run sits against the customer's spend cap::

    from agents import add_trace_processor
    from steadio_agents import SteadioTracingProcessor

    add_trace_processor(SteadioTracingProcessor(customer_id="acme", cap_cents=2_500))

``add_trace_processor`` runs alongside OpenAI's own backend;
``set_trace_processors([...])`` replaces it. Either works.

Scope, stated plainly: this processor is a **read-only observer of the local
run**. It does not enforce anything. Cap enforcement happens server-side at the
SteadIO gateway, which is what returns 402 ``spend_cap_exceeded`` when a capped
customer is over budget. The ``cap_state`` reported here is an in-process
estimate derived from the spans this processor has seen since it was
constructed -- useful for logging and for failing a run early, but not a
substitute for the gateway. Pass ``spend_lookup=`` to feed it authoritative
spend from your own records instead.

Contract notes from ``agents.tracing.processor_interface.TracingProcessor``:
handlers must be thread-safe, must return quickly, and must not raise into
agent execution. All three are honored below -- every hook is wrapped, and a
failure is logged and swallowed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen

from .pricing import DEFAULT_PRICING, PricingTable, cost_cents, format_cents

try:  # Use the real base class when the Agents SDK is installed.
    from agents.tracing.processor_interface import (  # type: ignore[import-not-found]
        TracingProcessor as _BaseTracingProcessor,
    )
except Exception:  # pragma: no cover - the SDK is an optional import here
    _BaseTracingProcessor = object  # type: ignore[assignment,misc]


logger = logging.getLogger("steadio.agents")

#: Cap states, least to most severe.
CAP_UNKNOWN = "unknown"
CAP_OK = "ok"
CAP_WARN = "warn"
CAP_OVER = "over"

# The Agents SDK normalizes provider usage onto input_tokens/output_tokens, but
# a raw OpenAI usage block uses prompt_tokens/completion_tokens. Accept both so
# a hand-built or passthrough span still attributes instead of silently
# recording zero.
_INPUT_KEYS = ("input_tokens", "prompt_tokens")
_OUTPUT_KEYS = ("output_tokens", "completion_tokens")


@dataclass
class ModelUsage:
    """Tokens attributed to one model within a single run."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunCost:
    """What one agent run (one trace) cost, and where it left the cap."""

    trace_id: str
    workflow_name: Optional[str] = None
    customer_id: Optional[str] = None
    by_model: Dict[str, ModelUsage] = field(default_factory=dict)
    #: Models seen with usage but absent from the pricing table; they
    #: contributed 0 cents. A non-empty list means the total is an undercount.
    unpriced_models: List[str] = field(default_factory=list)
    cost_cents: int = 0
    #: Cumulative in-process spend for this customer, including this run.
    spend_cents: int = 0
    cap_cents: Optional[int] = None
    cap_state: str = CAP_UNKNOWN

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        """One log-ready line."""
        who = self.customer_id or "unattributed"
        line = (
            f"[steadio] run {self.trace_id} customer={who} "
            f"tokens={self.total_tokens} cost={format_cents(self.cost_cents)}"
        )
        if self.cap_cents is not None:
            line += (
                f" spend={format_cents(self.spend_cents)}"
                f"/{format_cents(self.cap_cents)} cap={self.cap_state}"
            )
        if self.unpriced_models:
            line += f" unpriced={','.join(sorted(self.unpriced_models))}"
        return line


class SteadioTracingProcessor(_BaseTracingProcessor):  # type: ignore[misc,valid-type]
    """Aggregates per-run cost and cap state from Agents SDK trace spans.

    Args:
        customer_id: Default SteadIO end-customer this processor's runs belong
            to. A trace whose metadata carries ``customer_id`` overrides it.
        cap_cents: Monthly spend cap to compare against, in cents. Omit to
            report usage and cost without a cap verdict.
        warn_at: Fraction of the cap at which state becomes ``warn``.
        pricing: Override the model rate table (cents per 1M tokens).
        on_run: Called once per finished run with its ``RunCost``. Exceptions
            raised here are logged, never propagated.
        spend_lookup: ``(customer_id) -> spend_cents`` returning authoritative
            spend before this run. Use it to compare against SteadIO's ledger
            instead of this processor's in-process tally.
        log: Emit the one-line run summary at INFO. Defaults to True.
    """

    def __init__(
        self,
        *,
        customer_id: Optional[str] = None,
        cap_cents: Optional[int] = None,
        warn_at: float = 0.8,
        pricing: Optional[PricingTable] = None,
        on_run: Optional[Callable[[RunCost], None]] = None,
        spend_lookup: Optional[Callable[[Optional[str]], int]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        log: bool = True,
    ) -> None:
        self.customer_id = customer_id
        self.cap_cents = cap_cents
        self.warn_at = warn_at
        self.pricing = pricing if pricing is not None else DEFAULT_PRICING
        self.on_run = on_run
        self.log = log

        resolved_base = base_url or os.environ.get("STEADIO_BASE_URL") or os.environ.get("STEADIO_API_URL")
        resolved_key = api_key or os.environ.get("STEADIO_API_KEY")
        if spend_lookup is not None:
            self.spend_lookup = spend_lookup
        elif resolved_base and resolved_key:
            self.spend_lookup = _make_api_spend_lookup(resolved_base, resolved_key)
        else:
            self.spend_lookup = None

        self._lock = threading.Lock()
        self._active: Dict[str, RunCost] = {}
        self._spend_by_customer: Dict[Optional[str], int] = {}
        self._finished: List[RunCost] = []

    # --- public read surface -------------------------------------------------

    def spend_cents(self, customer_id: Optional[str] = None) -> int:
        """In-process spend tallied for a customer since construction."""
        with self._lock:
            return self._spend_by_customer.get(customer_id or self.customer_id, 0)

    def finished_runs(self) -> List[RunCost]:
        """Every run this processor has completed, oldest first."""
        with self._lock:
            return list(self._finished)

    # --- TracingProcessor hooks ---------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        try:
            trace_id = _trace_id(trace)
            if trace_id is None:
                return
            with self._lock:
                self._active[trace_id] = RunCost(
                    trace_id=trace_id,
                    workflow_name=getattr(trace, "name", None),
                    customer_id=_customer_from_trace(trace) or self.customer_id,
                    cap_cents=self.cap_cents,
                )
        except Exception:  # noqa: BLE001 - never break agent execution
            logger.exception("[steadio] on_trace_start failed")

    def on_span_start(self, span: Any) -> None:
        # Usage is only known once a span closes; nothing to do here.
        return None

    def on_span_end(self, span: Any) -> None:
        try:
            usage = _usage_of(span)
            if usage is None:
                return
            input_tokens, output_tokens = usage
            if input_tokens == 0 and output_tokens == 0:
                return
            trace_id = _trace_id(span)
            if trace_id is None:
                return
            model = _model_of(span) or "unknown"

            with self._lock:
                run = self._active.get(trace_id)
                if run is None:
                    # A span arriving without (or after) its trace still counts;
                    # opening a run here keeps the tally honest.
                    run = RunCost(
                        trace_id=trace_id,
                        customer_id=self.customer_id,
                        cap_cents=self.cap_cents,
                    )
                    self._active[trace_id] = run
                entry = run.by_model.get(model)
                if entry is None:
                    entry = ModelUsage(model=model)
                    run.by_model[model] = entry
                entry.input_tokens += input_tokens
                entry.output_tokens += output_tokens
        except Exception:  # noqa: BLE001
            logger.exception("[steadio] on_span_end failed")

    def on_trace_end(self, trace: Any) -> None:
        try:
            trace_id = _trace_id(trace)
            if trace_id is None:
                return
            self._finalize(trace_id)
        except Exception:  # noqa: BLE001
            logger.exception("[steadio] on_trace_end failed")

    def force_flush(self) -> None:
        """Finalize every run still open. Safe to call repeatedly."""
        try:
            with self._lock:
                open_ids = list(self._active)
            for trace_id in open_ids:
                self._finalize(trace_id)
        except Exception:  # noqa: BLE001
            logger.exception("[steadio] force_flush failed")

    def shutdown(self) -> None:
        """Flush, then drop all per-run state."""
        self.force_flush()
        with self._lock:
            self._active.clear()

    # --- internals -----------------------------------------------------------

    def _finalize(self, trace_id: str) -> None:
        with self._lock:
            run = self._active.pop(trace_id, None)
            if run is None:
                return

            total = 0
            for usage in run.by_model.values():
                cents = cost_cents(
                    usage.model, usage.input_tokens, usage.output_tokens, self.pricing
                )
                if cents == 0 and usage.total_tokens > 0:
                    run.unpriced_models.append(usage.model)
                total += cents
            run.cost_cents = total

            prior = self._spend_by_customer.get(run.customer_id, 0)
            if self.spend_lookup is not None:
                try:
                    prior = int(self.spend_lookup(run.customer_id))
                except Exception:  # noqa: BLE001 - fall back to the local tally
                    logger.exception("[steadio] spend_lookup failed")
            run.spend_cents = prior + total
            self._spend_by_customer[run.customer_id] = run.spend_cents
            run.cap_state = _cap_state(run.spend_cents, run.cap_cents, self.warn_at)
            self._finished.append(run)

        # Callbacks run outside the lock: user code must not be able to deadlock
        # the processor, and it may legitimately call back into it.
        if self.log:
            logger.info("%s", run.summary())
        if self.on_run is not None:
            try:
                self.on_run(run)
            except Exception:  # noqa: BLE001
                logger.exception("[steadio] on_run callback failed")


# --- API spend lookup --------------------------------------------------------


def _make_api_spend_lookup(base_url: str, api_key: str) -> Callable[[Optional[str]], int]:
    base = base_url.rstrip("/")

    def _lookup(customer_id: Optional[str]) -> int:
        if not customer_id:
            return 0
        url = f"{base}/customers/{quote(customer_id, safe='')}"
        req = Request(url, headers={"X-SteadIO-Key": api_key})
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return int(data["spend_cents"])

    return _lookup


# --- span/trace readers ------------------------------------------------------
#
# Read defensively via getattr: span data classes differ per span type
# (generation, response, turn, task) and the SDK adds new ones over time.


def _cap_state(spend_cents: int, cap_cents: Optional[int], warn_at: float) -> str:
    if cap_cents is None or cap_cents <= 0:
        return CAP_UNKNOWN
    if spend_cents >= cap_cents:
        return CAP_OVER
    if spend_cents >= cap_cents * warn_at:
        return CAP_WARN
    return CAP_OK


def _trace_id(obj: Any) -> Optional[str]:
    trace_id = getattr(obj, "trace_id", None)
    return trace_id if isinstance(trace_id, str) and trace_id else None


def _customer_from_trace(trace: Any) -> Optional[str]:
    metadata = getattr(trace, "metadata", None)
    if isinstance(metadata, Mapping):
        value = metadata.get("customer_id") or metadata.get("x-customer-id")
        if isinstance(value, str) and value:
            return value
    return None


def _model_of(span: Any) -> Optional[str]:
    data = getattr(span, "span_data", None)
    if data is None:
        return None
    model = getattr(data, "model", None)
    if isinstance(model, str) and model:
        return model
    # ResponseSpanData carries the model on the response object instead.
    response = getattr(data, "response", None)
    model = getattr(response, "model", None)
    return model if isinstance(model, str) and model else None


def _usage_of(span: Any) -> Optional[tuple]:
    data = getattr(span, "span_data", None)
    if data is None:
        return None
    usage = getattr(data, "usage", None)
    if not isinstance(usage, Mapping):
        return None
    return (_first_int(usage, _INPUT_KEYS), _first_int(usage, _OUTPUT_KEYS))


def _first_int(source: Mapping[str, Any], keys: tuple) -> int:
    for key in keys:
        value = source.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
    return 0
