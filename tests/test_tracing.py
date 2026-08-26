"""Tests for SteadioTracingProcessor.

The doubles below mirror the real shapes in
``agents.tracing.span_data`` / ``agents.tracing.traces`` (verified against
openai/openai-agents-python@main): a generation span exposes
``span_data.model`` and ``span_data.usage`` with ``input_tokens`` /
``output_tokens``; a response span carries the model on ``span_data.response``.
They are inputs to the processor, not stand-ins for it -- every assertion is
about the processor's own aggregation, cost math, and cap logic.

Run: python3 -m unittest discover -s integrations/openai-agents -t .
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steadio_agents import (  # noqa: E402
    CAP_OK,
    CAP_OVER,
    CAP_UNKNOWN,
    CAP_WARN,
    SteadioTracingProcessor,
)
from steadio_agents.pricing import cost_cents  # noqa: E402


# --- doubles -----------------------------------------------------------------


class GenerationSpanData:
    type = "generation"

    def __init__(self, model=None, usage=None):
        self.model = model
        self.usage = usage


class ResponseSpanData:
    type = "response"

    def __init__(self, response=None, usage=None):
        self.response = response
        self.usage = usage


class Response:
    def __init__(self, model):
        self.model = model


class FunctionSpanData:
    """A span type that carries no usage at all."""

    type = "function"

    def __init__(self, name):
        self.name = name


class Span:
    def __init__(self, trace_id, span_data, span_id="span_1"):
        self.trace_id = trace_id
        self.span_id = span_id
        self.span_data = span_data


class Trace:
    def __init__(self, trace_id, name=None, metadata=None):
        self.trace_id = trace_id
        self.name = name
        self.metadata = metadata


def gen(trace_id, model, input_tokens, output_tokens):
    return Span(
        trace_id,
        GenerationSpanData(
            model=model, usage={"input_tokens": input_tokens, "output_tokens": output_tokens}
        ),
    )


def run_trace(processor, trace, spans):
    """Drive one full trace through the processor, in SDK call order."""
    processor.on_trace_start(trace)
    for span in spans:
        processor.on_span_start(span)
        processor.on_span_end(span)
    processor.on_trace_end(trace)


# --- cost math ---------------------------------------------------------------


class CostMathTest(unittest.TestCase):
    """Ports of the cost engine's own worked examples (cents per 1M, ceil)."""

    def test_gpt_4o_worked_example(self):
        # (12000/1e6)*250 + (3500/1e6)*1000 = 3.0 + 3.5 = 6.5 -> ceil 7
        self.assertEqual(cost_cents("gpt-4o", 12_000, 3_500), 7)

    def test_ceil_bias_rounds_a_sub_cent_call_up_to_one(self):
        # (1000/1e6)*15 + (200/1e6)*60 = 0.027 -> ceil 1
        self.assertEqual(cost_cents("gpt-4o-mini", 1_000, 200), 1)

    def test_unknown_model_is_free_not_an_error(self):
        self.assertEqual(cost_cents("some-model-we-have-never-seen", 10_000, 10_000), 0)

    def test_zero_tokens_is_zero_cents(self):
        self.assertEqual(cost_cents("gpt-4o", 0, 0), 0)


# --- aggregation -------------------------------------------------------------


class AggregationTest(unittest.TestCase):
    def test_sums_usage_across_spans_in_one_run(self):
        p = SteadioTracingProcessor(log=False)
        run_trace(
            p,
            Trace("trace_a", name="support-agent"),
            [gen("trace_a", "gpt-4o", 10_000, 2_000), gen("trace_a", "gpt-4o", 2_000, 1_500)],
        )

        (run,) = p.finished_runs()
        self.assertEqual(run.input_tokens, 12_000)
        self.assertEqual(run.output_tokens, 3_500)
        self.assertEqual(run.total_tokens, 15_500)
        self.assertEqual(run.workflow_name, "support-agent")
        # Same totals as the worked example -> same 7 cents.
        self.assertEqual(run.cost_cents, 7)

    def test_costs_each_model_separately_then_sums(self):
        p = SteadioTracingProcessor(log=False)
        run_trace(
            p,
            Trace("trace_b"),
            [gen("trace_b", "gpt-4o", 12_000, 3_500), gen("trace_b", "gpt-4o-mini", 1_000, 200)],
        )

        (run,) = p.finished_runs()
        self.assertEqual(sorted(run.by_model), ["gpt-4o", "gpt-4o-mini"])
        # Per-model ceil, then sum: 7 + 1. Not ceil of the combined 6.527.
        self.assertEqual(run.cost_cents, 8)

    def test_reads_model_off_a_response_span(self):
        p = SteadioTracingProcessor(log=False)
        span = Span(
            "trace_c",
            ResponseSpanData(
                response=Response("gpt-4o"),
                usage={"input_tokens": 12_000, "output_tokens": 3_500},
            ),
        )
        run_trace(p, Trace("trace_c"), [span])

        (run,) = p.finished_runs()
        self.assertEqual(run.cost_cents, 7)
        self.assertIn("gpt-4o", run.by_model)

    def test_accepts_raw_openai_usage_key_names(self):
        p = SteadioTracingProcessor(log=False)
        span = Span(
            "trace_d",
            GenerationSpanData(
                model="gpt-4o",
                usage={"prompt_tokens": 12_000, "completion_tokens": 3_500},
            ),
        )
        run_trace(p, Trace("trace_d"), [span])

        (run,) = p.finished_runs()
        self.assertEqual(run.input_tokens, 12_000)
        self.assertEqual(run.cost_cents, 7)

    def test_ignores_spans_with_no_usage(self):
        p = SteadioTracingProcessor(log=False)
        run_trace(
            p,
            Trace("trace_e"),
            [
                Span("trace_e", FunctionSpanData("lookup_order")),
                Span("trace_e", GenerationSpanData(model="gpt-4o", usage=None)),
                gen("trace_e", "gpt-4o", 12_000, 3_500),
            ],
        )

        (run,) = p.finished_runs()
        self.assertEqual(run.by_model["gpt-4o"].input_tokens, 12_000)
        self.assertEqual(run.cost_cents, 7)

    def test_flags_unpriced_models_so_the_total_is_not_read_as_complete(self):
        p = SteadioTracingProcessor(log=False)
        run_trace(p, Trace("trace_f"), [gen("trace_f", "gpt-5-imaginary", 50_000, 50_000)])

        (run,) = p.finished_runs()
        self.assertEqual(run.cost_cents, 0)
        self.assertEqual(run.unpriced_models, ["gpt-5-imaginary"])

    def test_runs_are_isolated_from_each_other(self):
        p = SteadioTracingProcessor(log=False)
        run_trace(p, Trace("trace_g"), [gen("trace_g", "gpt-4o", 12_000, 3_500)])
        run_trace(p, Trace("trace_h"), [gen("trace_h", "gpt-4o-mini", 1_000, 200)])

        first, second = p.finished_runs()
        self.assertEqual(first.cost_cents, 7)
        self.assertEqual(second.cost_cents, 1)
        self.assertEqual(first.total_tokens, 15_500)
        self.assertEqual(second.total_tokens, 1_200)


# --- cap state ---------------------------------------------------------------


class CapStateTest(unittest.TestCase):
    def test_no_cap_configured_reports_unknown(self):
        p = SteadioTracingProcessor(log=False)
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        self.assertEqual(p.finished_runs()[0].cap_state, CAP_UNKNOWN)

    def test_well_under_cap_is_ok(self):
        p = SteadioTracingProcessor(cap_cents=100, log=False)
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])  # 7 cents
        run = p.finished_runs()[0]
        self.assertEqual(run.spend_cents, 7)
        self.assertEqual(run.cap_state, CAP_OK)

    def test_crossing_the_warn_threshold_across_runs(self):
        # cap 10c, warn at 80% => 8c. Two 7c runs: 7 -> ok, 14 -> over.
        p = SteadioTracingProcessor(cap_cents=10, log=False)
        run_trace(p, Trace("t1"), [gen("t1", "gpt-4o", 12_000, 3_500)])
        run_trace(p, Trace("t2"), [gen("t2", "gpt-4o", 12_000, 3_500)])

        first, second = p.finished_runs()
        self.assertEqual((first.spend_cents, first.cap_state), (7, CAP_OK))
        self.assertEqual((second.spend_cents, second.cap_state), (14, CAP_OVER))

    def test_warn_state_is_reachable(self):
        # cap 8c, warn at 80% => 6.4c. A single 7c run lands in warn, not over.
        p = SteadioTracingProcessor(cap_cents=8, log=False)
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        self.assertEqual(p.finished_runs()[0].cap_state, CAP_WARN)

    def test_spend_is_tallied_per_customer_not_globally(self):
        p = SteadioTracingProcessor(cap_cents=100, log=False)
        run_trace(
            p,
            Trace("t1", metadata={"customer_id": "acme"}),
            [gen("t1", "gpt-4o", 12_000, 3_500)],
        )
        run_trace(
            p,
            Trace("t2", metadata={"customer_id": "globex"}),
            [gen("t2", "gpt-4o", 12_000, 3_500)],
        )
        run_trace(
            p,
            Trace("t3", metadata={"customer_id": "acme"}),
            [gen("t3", "gpt-4o", 12_000, 3_500)],
        )

        self.assertEqual(p.spend_cents("acme"), 14)
        self.assertEqual(p.spend_cents("globex"), 7)

    def test_trace_metadata_customer_overrides_the_default(self):
        p = SteadioTracingProcessor(customer_id="default-co", log=False)
        run_trace(
            p, Trace("t", metadata={"customer_id": "acme"}), [gen("t", "gpt-4o", 12_000, 3_500)]
        )
        self.assertEqual(p.finished_runs()[0].customer_id, "acme")

    def test_spend_lookup_supplies_authoritative_prior_spend(self):
        p = SteadioTracingProcessor(
            customer_id="acme", cap_cents=100, spend_lookup=lambda _c: 95, log=False
        )
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])

        run = p.finished_runs()[0]
        self.assertEqual(run.spend_cents, 102)  # 95 authoritative + 7 this run
        self.assertEqual(run.cap_state, CAP_OVER)

    def test_a_failing_spend_lookup_falls_back_to_the_local_tally(self):
        def boom(_customer):
            raise RuntimeError("ledger unreachable")

        p = SteadioTracingProcessor(cap_cents=100, spend_lookup=boom, log=False)
        with self.assertLogs("steadio.agents", level="ERROR"):
            run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])

        self.assertEqual(p.finished_runs()[0].spend_cents, 7)


# --- lifecycle and resilience ------------------------------------------------


class LifecycleTest(unittest.TestCase):
    def test_on_run_fires_once_per_completed_run(self):
        seen = []
        p = SteadioTracingProcessor(on_run=seen.append, log=False)
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])

        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].cost_cents, 7)

    def test_on_run_does_not_fire_before_the_trace_ends(self):
        seen = []
        p = SteadioTracingProcessor(on_run=seen.append, log=False)
        trace = Trace("t")
        p.on_trace_start(trace)
        p.on_span_end(gen("t", "gpt-4o", 12_000, 3_500))
        self.assertEqual(seen, [])

        p.on_trace_end(trace)
        self.assertEqual(len(seen), 1)

    def test_force_flush_finalizes_a_run_left_open(self):
        p = SteadioTracingProcessor(log=False)
        p.on_trace_start(Trace("t"))
        p.on_span_end(gen("t", "gpt-4o", 12_000, 3_500))
        self.assertEqual(p.finished_runs(), [])

        p.force_flush()
        self.assertEqual(len(p.finished_runs()), 1)
        self.assertEqual(p.finished_runs()[0].cost_cents, 7)

    def test_force_flush_is_idempotent(self):
        p = SteadioTracingProcessor(log=False)
        p.on_trace_start(Trace("t"))
        p.on_span_end(gen("t", "gpt-4o", 12_000, 3_500))
        p.force_flush()
        p.force_flush()
        self.assertEqual(len(p.finished_runs()), 1)

    def test_trace_end_after_flush_does_not_double_count(self):
        p = SteadioTracingProcessor(log=False)
        trace = Trace("t")
        p.on_trace_start(trace)
        p.on_span_end(gen("t", "gpt-4o", 12_000, 3_500))
        p.force_flush()
        p.on_trace_end(trace)

        self.assertEqual(len(p.finished_runs()), 1)
        self.assertEqual(p.spend_cents(), 7)

    def test_shutdown_flushes_then_clears_active_state(self):
        p = SteadioTracingProcessor(log=False)
        p.on_trace_start(Trace("t"))
        p.on_span_end(gen("t", "gpt-4o", 12_000, 3_500))
        p.shutdown()

        self.assertEqual(len(p.finished_runs()), 1)
        p.shutdown()  # safe to call again
        self.assertEqual(len(p.finished_runs()), 1)

    def test_a_span_arriving_without_its_trace_still_counts(self):
        p = SteadioTracingProcessor(log=False)
        p.on_span_end(gen("orphan", "gpt-4o", 12_000, 3_500))
        p.force_flush()

        (run,) = p.finished_runs()
        self.assertEqual(run.trace_id, "orphan")
        self.assertEqual(run.cost_cents, 7)

    def test_a_raising_callback_never_reaches_the_agent(self):
        def boom(_run):
            raise RuntimeError("user callback exploded")

        p = SteadioTracingProcessor(on_run=boom, log=False)
        with self.assertLogs("steadio.agents", level="ERROR"):
            run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])

        # The run still completed and was recorded.
        self.assertEqual(len(p.finished_runs()), 1)

    def test_malformed_spans_and_traces_are_survivable(self):
        p = SteadioTracingProcessor(log=False)
        for bad in (None, object(), Span(None, None), Trace(None)):
            p.on_trace_start(bad)
            p.on_span_start(bad)
            p.on_span_end(bad)
            p.on_trace_end(bad)

        # Nothing recorded, nothing raised; a good run afterwards still works.
        self.assertEqual(p.finished_runs(), [])
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        self.assertEqual(len(p.finished_runs()), 1)

    def test_concurrent_traces_do_not_lose_or_cross_attribute_usage(self):
        p = SteadioTracingProcessor(log=False)
        trace_ids = [f"t{i}" for i in range(16)]

        def drive(trace_id):
            run_trace(
                p,
                Trace(trace_id),
                [gen(trace_id, "gpt-4o", 12_000, 3_500) for _ in range(4)],
            )

        threads = [threading.Thread(target=drive, args=(tid,)) for tid in trace_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        runs = p.finished_runs()
        self.assertEqual(len(runs), 16)
        self.assertEqual({r.trace_id for r in runs}, set(trace_ids))
        for run in runs:
            self.assertEqual(run.input_tokens, 48_000)
            self.assertEqual(run.output_tokens, 14_000)
        # 4x the worked example per run: 26 cents each.
        self.assertEqual(p.spend_cents(), 16 * 26)


class SummaryTest(unittest.TestCase):
    def test_summary_renders_cost_in_dollars_never_bare_cents(self):
        p = SteadioTracingProcessor(customer_id="acme", cap_cents=2_500, log=False)
        run_trace(p, Trace("t", name="support"), [gen("t", "gpt-4o", 12_000, 3_500)])

        summary = p.finished_runs()[0].summary()
        self.assertIn("customer=acme", summary)
        self.assertIn("cost=$0.07", summary)
        self.assertIn("$0.07/$25.00", summary)
        self.assertIn("cap=ok", summary)

    def test_summary_marks_an_unattributed_run(self):
        p = SteadioTracingProcessor(log=False)
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        self.assertIn("customer=unattributed", p.finished_runs()[0].summary())


# --- API-backed spend lookup -------------------------------------------------


class APISpendLookupTest(unittest.TestCase):
    def _mock_urlopen(self, response_body, status=200):
        mock_response = MagicMock()
        mock_response.status = status
        mock_response.read.return_value = json.dumps(response_body).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        return mock_response

    @patch("steadio_agents.tracing.urlopen")
    def test_uses_api_spend_lookup_when_credentials_provided(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            {"customer_id": "acme", "cap_cents": 2500, "spend_cents": 800, "enforcement_mode": "block", "state": "ok"}
        )
        p = SteadioTracingProcessor(
            customer_id="acme", cap_cents=2500,
            base_url="https://api.steadio.ai/v1", api_key="sk-ste-test", log=False,
        )
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        run = p.finished_runs()[0]
        self.assertEqual(run.spend_cents, 807)
        mock_urlopen.assert_called_once()

    @patch("steadio_agents.tracing.urlopen")
    def test_falls_back_to_local_tally_on_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = OSError("connection refused")
        p = SteadioTracingProcessor(
            customer_id="acme", cap_cents=2500,
            base_url="https://api.steadio.ai/v1", api_key="sk-ste-test", log=False,
        )
        with self.assertLogs("steadio.agents", level="ERROR"):
            run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        run = p.finished_runs()[0]
        self.assertEqual(run.spend_cents, 7)

    def test_no_api_lookup_without_credentials(self):
        p = SteadioTracingProcessor(customer_id="acme", cap_cents=2500, log=False)
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        self.assertEqual(p.finished_runs()[0].spend_cents, 7)

    @patch("steadio_agents.tracing.urlopen")
    def test_explicit_spend_lookup_overrides_api_lookup(self, mock_urlopen):
        p = SteadioTracingProcessor(
            customer_id="acme", cap_cents=2500,
            base_url="https://api.steadio.ai/v1", api_key="sk-ste-test",
            spend_lookup=lambda _c: 50, log=False,
        )
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        self.assertEqual(p.finished_runs()[0].spend_cents, 57)
        mock_urlopen.assert_not_called()

    @patch("steadio_agents.tracing.urlopen")
    def test_api_lookup_sends_correct_headers_and_path(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            {"customer_id": "acme", "cap_cents": 2500, "spend_cents": 100, "enforcement_mode": "block", "state": "ok"}
        )
        p = SteadioTracingProcessor(
            customer_id="acme", cap_cents=2500,
            base_url="https://api.steadio.ai/v1", api_key="sk-ste-test", log=False,
        )
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "https://api.steadio.ai/v1/customers/acme")
        self.assertEqual(req.get_header("X-steadio-key"), "sk-ste-test")

    @patch("steadio_agents.tracing.urlopen")
    def test_api_lookup_url_encodes_customer_id(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_urlopen(
            {"customer_id": "a/b?c", "cap_cents": 2500, "spend_cents": 0, "enforcement_mode": "block", "state": "ok"}
        )
        p = SteadioTracingProcessor(
            customer_id="a/b?c", cap_cents=2500,
            base_url="https://api.steadio.ai/v1", api_key="sk-ste-test", log=False,
        )
        run_trace(p, Trace("t"), [gen("t", "gpt-4o", 12_000, 3_500)])
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "https://api.steadio.ai/v1/customers/a%2Fb%3Fc")


if __name__ == "__main__":
    unittest.main()
