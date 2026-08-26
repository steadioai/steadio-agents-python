# SteadIO for the OpenAI Agents SDK

Per-run cost and spend-cap visibility for agent runs, as a
[`TracingProcessor`](https://openai.github.io/openai-agents-python/ref/tracing/processors/).

Every agent run reports how many tokens it burned, what that cost in cents, and
where the run left that customer against their spend cap.

```
[steadio] run trace_abc123 customer=acme tokens=15500 cost=$0.07 spend=$21.40/$25.00 cap=warn
```

## Install

```bash
pip install steadio-agents
```

> Not yet on PyPI. Until the first release lands, install from a checkout:
> `pip install ./integrations/openai-agents`

## Use

```python
from agents import Agent, Runner, add_trace_processor
from steadio_agents import SteadioTracingProcessor

add_trace_processor(SteadioTracingProcessor(customer_id="acme", cap_cents=2_500))

agent = Agent(name="Support", instructions="Help the customer.")
Runner.run_sync(agent, "Where is my order?")
```

`add_trace_processor` runs alongside OpenAI's own tracing backend.
`set_trace_processors([SteadioTracingProcessor(...)])` replaces it instead.

### Per-customer attribution

If one process serves many of your customers, set the customer per run instead
of per processor — trace metadata wins over the processor default:

```python
from agents import trace

with trace("support-run", metadata={"customer_id": "acme"}):
    Runner.run_sync(agent, "Where is my order?")
```

### Acting on the result

`on_run` fires once per finished run with a `RunCost`:

```python
def guard(run):
    if run.cap_state == "over":
        raise RuntimeError(f"{run.customer_id} is over cap: {run.summary()}")

add_trace_processor(SteadioTracingProcessor(cap_cents=2_500, on_run=guard))
```

`RunCost` carries `cost_cents`, `spend_cents`, `cap_cents`, `cap_state`
(`unknown` / `ok` / `warn` / `over`), `input_tokens`, `output_tokens`,
`by_model`, and `unpriced_models`.

## Enforcing caps, not just observing them

This processor observes. It does not enforce — it sees spans after the fact and
cannot stop a call.

Enforcement happens at the SteadIO gateway. Point the SDK's client at it and
your own provider key is forwarded upstream per request:

```python
from agents import set_default_openai_client
from openai import AsyncOpenAI

set_default_openai_client(AsyncOpenAI(
    base_url="https://api.steadio.ai/v1",
    api_key="<your OpenAI key>",       # forwarded upstream, not stored
    default_headers={
        "X-SteadIO-Key": "st_...",      # your SteadIO key
        "X-Customer-Id": "acme",        # who this spend belongs to
    },
))
```

Customers auto-appear with month-to-date spend on first sight of an
`X-Customer-Id`. A customer whose monthly cap is set to block gets a
machine-readable `402` **before** the provider is called:

```json
{
  "error": "spend_cap_exceeded",
  "customer": "acme",
  "cap_cents": 2500,
  "spend_cents": 2531
}
```

Cap **blocking** requires a paid plan; the free tier is monitor-only, so caps
record overage without turning customers away. Plans are $0 / $29 / $79 / $199
per month by customer count.

Run both together and the gateway is the enforcement point while the processor
gives you the same numbers inside your own logs and traces.

## What the numbers mean

- **Cost is computed locally** from span usage, using SteadIO's convention:
  rates in cents per 1M tokens, per-model, rounded up to whole cents per model.
  A run's cost is the sum of its per-model costs.
- **`cap_state` is an in-process estimate.** By default the processor tallies
  only the spend it has seen since it was constructed — not your account's
  month-to-date total. Pass `spend_lookup=` to supply authoritative prior spend
  from your own records:

  ```python
  SteadioTracingProcessor(cap_cents=2_500, spend_lookup=lambda customer: my_ledger[customer])
  ```

- **Unknown models cost 0 and are listed in `unpriced_models`.** A non-empty
  list means the run's total is an undercount, not that the run was free. The
  bundled rate table is a small floor covering common OpenAI, Anthropic, and
  Gemini models; pass `pricing=` to extend or override it.
- **SteadIO's own ledger is authoritative.** Requests routed through the
  gateway are metered there for both streaming and non-streaming traffic; this
  processor is a local read of the same run.

## Behavior under failure

The SDK requires processors to be thread-safe, quick, and non-throwing. All
three hold here: every hook is wrapped, and a failure inside the processor or
inside your `on_run` callback is logged to the `steadio.agents` logger and
swallowed rather than raised into agent execution.

`force_flush()` finalizes runs still open; `shutdown()` flushes and clears.

## Tests

```bash
python3 -m unittest discover -s integrations/openai-agents/tests -t integrations/openai-agents/tests
```

## License

MIT
