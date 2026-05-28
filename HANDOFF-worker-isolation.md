# HANDOFF — `LLAMA_WORKER_ISOLATION` + admin endpoints (GO-LIVE-PLAN §3.4.b)

**Status:** **NOT in this commit.** The kobbler `kob-gpu-gate` already
registers an LLM heavy service with `unload_path: "/v1/admin/unload"` +
`drain_path: "/v1/admin/drain"`, and `acquire_for_avatar` already does the
drain → wait-for-in-flight → unload dance against those paths. So as soon as
this fork ships the endpoints, the avatar↔LLM coexistence story works without
any further kobbler-side change.

Today: koblem-llama is `docker.oldug.com/koblem-llama:latest` (the
TurboQuant fork). The Avatar render path **silently no-ops** when the LLM is
unregistered or when the endpoints aren't there — fine for v0 since the box
the agent verified on doesn't currently run llama-server. But ship this for
real two-heavies coexistence.

## What it needs to do

Per GO-LIVE-PLAN §3.4.b:

- Env-gated `LLAMA_WORKER_ISOLATION=1`. When set, on the first chat request,
  `fork()` a child that holds the CUDA context + the GGUF + the spec-mtp
  draft + mmproj. The parent stays CUDA-free and owns the HTTP port + the
  in-flight counter.
- `POST /v1/admin/unload` — SIGKILL the child. Kernel exit tears down the
  CUDA context → ALL VRAM reclaimed. Idempotent: an idle parent returns
  200 with `{status: "idle"}`.
- `POST /v1/admin/drain` — set a flag that 503's *new* `/v1/chat/completions`
  + `/v1/completions` requests. Existing in-flight streams run to `[DONE]`
  naturally. Clearing the flag happens via `POST /v1/admin/load` (or
  automatically on next acquire-for-llm, but explicit-clear is cleaner).
- `POST /v1/admin/load` — optional pre-warm. If the child is dead, fork +
  load eagerly so the next chat request doesn't pay the cold reload tax.
- `GET /health` — augment the existing response with `{loaded, busy,
  draining, in_flight}` so the kob-gpu-gate `service_states()` shape and
  the koblem `/api/v1/gpu/services` UI can show LLM state.

## The non-trivial bits (cribbed from GO-LIVE-PLAN §3.4.b)

- **SSE relay across the IPC boundary.** Chat completions stream
  `text/event-stream` — the parent has to forward the child's SSE frames
  out the inbound HTTP socket without re-encoding. Two approaches:

  - **dup'd-fd / SCM_RIGHTS** — child sends its write-end fd to the parent
    over a Unix-domain socket; parent splices fd → response socket. Lowest
    overhead.
  - **Per-request named pipe** — parent opens a FIFO before sending the
    request blob to the child; reads SSE frames from the FIFO, writes to
    the response. Simpler, marginally slower.

- **Drain semantics under streams.** `/v1/admin/drain` blocks *new*
  requests only. The currently-streaming response runs to its `[DONE]`
  event. Keep-alive connections that haven't sent a request yet stay open
  (their next request gets the 503).

- **Cold-reload latency.** ~5 s warm-page-cache, < 2 s if `/models` is
  already paged. First chat after an avatar render eats this — acceptable
  (the kid pressing Avatar then immediately resuming chat is the expected
  flow, and 5 s is invisible compared to the avatar wall they just sat
  through).

## Existing sibling references (read these BEFORE writing)

1. `~/dev/qwen3-tts.cpp/src/server.cpp` (THE blueprint):
   - L552 — `fork()` on first request when `QWEN3_TTS_WORKER_ISOLATION=1`.
   - L856–880 — request marshalling to child over a Unix-domain socket.
   - L1152–1158 — `POST /v1/admin/unload` handler.

2. `~/dev/siglip2.cpp/src/{siglip2_server.cpp, worker_ipc.cpp, worker_session.cpp}`
   — second worker-isolation reference, slightly different IPC shape.

3. `~/dev/parakeet.cpp/examples/server/server.cpp` — third reference;
   useful for the "`/unload` returns clean even when no model is loaded"
   idempotency pattern (compose's healthcheck depends on this).

## Where to make the change in this fork

The server entry is at `examples/server/server.cpp` (or the standard
llama.cpp server path — check the latest sync point from upstream). Add a
top-of-main fork helper + an IPC channel module + the three admin route
handlers. Bind the existing chat-completion handler behind a
`forward_to_child(request, response)` shim.

## Verification on this box

The `llama-server-dev` profile (`docker/llama-server-dev/` + `LLAMA_SRC=
$HOME/dev/llama.cpp-turboquant`) builds against the local source tree on
port 8093. End-to-end test:

1. `docker compose --profile llama-dev up -d llama-server-dev`
2. Send a chat completion, watch VRAM go up (~6.5 GB).
3. `POST localhost:8093/v1/admin/drain`, fire another chat → expect 503.
4. `POST localhost:8093/v1/admin/unload`, watch VRAM return to 149 MiB.
5. Send another chat — expect cold load (~5 s) then normal streaming.

Run the avatar render concurrently to verify the full
`acquire_for_avatar` drain dance (kobbler must point `LLAMA_SERVER_URL` at
port 8093 for this test).

## After it ships

1. Owner triggers a registry rebuild of `docker.oldug.com/koblem-llama:latest`.
2. `docker compose pull llama-server && docker compose up -d llama-server`
   on this box.
3. `kob-gpu-gate` test `avatar_drains_then_unloads_llm` (already landed)
   becomes load-bearing — the drain dance will fire in prod.
4. The `is_avatar_busy` 503-on-chat path becomes the prod failure mode.

## Don't get clever

This is **port the qwen3-tts.cpp shape**, not "design a new IPC". The
shape works; the cost of inventing something different is real and the
ROI is zero.
