# CacheWise: KV Cache Management for Coding Agents

## Introduction

Coding agents (e.g. Claude Code) run long, closed-loop sessions: each LLM
request extends a growing conversation prefix, and between requests the agent
executes tool calls (`Bash`, `Read`, `Edit`, ...) whose durations vary by
orders of magnitude. vLLM's default policies are workload-agnostic: requests
are admitted FCFS, and cached KV blocks are evicted in LRU order, which under
memory pressure repeatedly evicts prefixes that an active session is about to
reuse.

CacheWise adds two independent, opt-in mechanisms tuned for this workload:

- **Prefix-aware request scheduling** (`--scheduling-policy prefix_aware`):
  among waiting requests, dispatch the one that needs the fewest additional
  KV cache blocks (i.e. the highest prefix cache overlap), instead of FCFS.
- **Predictive KV cache eviction**
  (`--kv-cache-eviction-policy predictive`): instead of LRU, evict cached
  blocks in order of *predicted time-to-next-reuse*. The engine tracks agent
  sessions by their block hash chains, parses the tool calls each response
  ends with, and learns per-tool duration distributions online from observed
  (tool call → next request) gaps. Blocks of a session that just launched a
  long-running tool (or is waiting on a human) are evicted first; blocks of a
  session about to return are protected.

No client cooperation is required: sessions are identified purely from
prefix-cache block hashes, and tool calls are inferred from model output via
the configured tool parser. This works with unmodified agents talking to the
OpenAI-compatible (`/v1/chat/completions`) or Anthropic-compatible
(`/v1/messages`) endpoints.

## Usage

```bash
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
    --kv-cache-eviction-policy predictive \
    --scheduling-policy prefix_aware \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

The two flags compose but are independent; either can be enabled alone.
Tool-call inference requires `--tool-call-parser`; without it, responses are
treated as turns with no tool call (the "human pause" category), which still
captures the dominant idle pattern.

### Options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--kv-cache-eviction-policy` | `lru` | `lru` or `predictive`. |
| `--cachewise-rebuild-interval` | `3` | Engine iterations between eviction-order rebuilds. |
| `--cachewise-session-ttl` | `1800` | Seconds before an idle session is dropped from tracking. |
| `--cachewise-predictor` | `tool_name` | `tool_name` or `tfidf_kmeans` (clusters tool arguments; requires scikit-learn). |
| `--cachewise-bootstrap-path` | unset | JSONL file of historical `{"tool", "args", "duration_s"}` samples to warm-start the predictor. |
| `--cachewise-default-reuse-s` | `120` | Predicted reuse time for blocks without session metadata. |
| `--scheduling-policy prefix_aware` | `fcfs` | Enable prefix-aware scheduling. |
| `--prefix-aware-max-wait-s` | `30` | Anti-starvation: a request waiting longer is dispatched FCFS. |
| `--prefix-aware-max-candidates` | `64` | Oldest waiting requests scored per scheduling decision. |

### Ground-truth hints

For trace replay and ablation studies, a client may pass the tool calls it is
about to execute alongside a request; the hint takes precedence over parsed
tool calls for that turn:

```python
client.chat.completions.create(
    model=...,
    messages=...,
    extra_body={
        "vllm_xargs": {
            "cachewise_next_tools": '[{"name": "Bash", "args": "pytest -x"}]'
        }
    },
)
```

### Composing with KV cache offloading

Predictive eviction composes with KV cache offloading
(`--kv-offloading-size`): eviction candidates are exposed to offloading
backends in predicted-reuse order, so proactive offload copies out the
blocks whose sessions are furthest from returning first, and eviction of
already-offloaded blocks stays cheap.

## Limitations

- Predictive eviction requires prefix caching and currently supports models
  with a single full-attention KV cache group; hybrid/SWA models fall back
  to LRU with a warning.
- Under prefix-aware scheduling, per-request `priority` is ignored, and P99
  request latency can increase: the policy optimizes end-to-end session
  completion time rather than request tail latency.
- Requests with `n > 1` are excluded from tool-call reporting.
