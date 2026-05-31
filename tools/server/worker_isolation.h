// worker_isolation.h — `LLAMA_WORKER_ISOLATION=1` parent/child split for
// llama-server, matching the shape used in qwen3-tts.cpp + siglip2.cpp.
//
// PARENT (no env var change after this point):
//   - skips llama_backend_init() / model load → CUDA primary context is
//     NEVER created in this process. nvidia-smi shows ~0 MiB for the
//     parent's pid.
//   - owns the inbound HTTP port (params.port).
//   - on the first /v1/chat/completions (or /v1/completions, or
//     /v1/embeddings) request, fork+execv's its own argv with
//     LLAMA_WORKER_ISOLATION cleared + LLAMA_WORKER_ISOLATION_CHILD=1 set,
//     listening on a private localhost port. Subsequent requests are
//     proxied to that child via server_http_proxy (same primitive the
//     router server already uses — gives us streaming SSE relay for free).
//   - tracks an in-flight chat counter and a "draining" flag.
//   - registers /v1/admin/{drain,load,unload} that operate on the child.
//
// CHILD:
//   - sees LLAMA_WORKER_ISOLATION_CHILD=1, skips this code entirely, runs
//     as a normal single-process llama-server on the private port.
//
// SSE relay:
//   - parent uses server_http_proxy → cpp-httplib request with a
//     ContentReceiverWithProgress callback. Each chunk written by the
//     child is enqueued in a pipe_t and re-emitted from the parent's
//     chunked content provider. No re-encoding. Works for both
//     streaming (text/event-stream) and non-streaming JSON responses
//     because httplib uses the same content_receiver path for both.
//
// Drain semantics:
//   - drain=true causes /v1/{chat,}completions + /v1/embeddings + /v1/rerank
//     to 503 on entry. Existing in-flight proxy streams keep running —
//     the drain flag is only checked at handler entry, not inside the
//     proxy. POST /v1/admin/load clears the drain flag.

#pragma once

#include "server-http.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct subprocess_s;
struct common_params;

class worker_isolation {
public:
    worker_isolation(const common_params & params, int argc, char ** argv);
    ~worker_isolation();

    // Spawn child + poll its /health until ready (cold reload = ~5-10s).
    // Returns true on success. Caller MUST hold no other lock when
    // entering this — internally locks for the spawn duration.
    bool ensure_loaded();

    // SIGKILL the child + reap. Idempotent: returns false if already
    // idle, true if a live child was killed.
    bool shutdown_child();

    // Set / read drain flag. Drain=true 503's NEW requests; in-flight
    // proxy streams keep going.
    void set_draining(bool d) { draining_.store(d); }
    bool is_draining() const   { return draining_.load(); }

    bool is_loaded() const;
    int  in_flight() const { return in_flight_.load(); }

    // RAII guard incremented around proxied requests; the parent's
    // /health reports `in_flight` for the kob-gpu-gate drain loop.
    struct in_flight_guard {
        worker_isolation * self;
        in_flight_guard(worker_isolation * w) : self(w) { if (self) self->in_flight_.fetch_add(1); }
        ~in_flight_guard()                              { if (self) self->in_flight_.fetch_sub(1); }
        in_flight_guard(const in_flight_guard &) = delete;
        in_flight_guard & operator=(const in_flight_guard &) = delete;
    };

    // Proxy a request to the child. The returned response is either a
    // streaming wrapper (server_http_proxy) or an error response. On
    // first call, lazily spawns the child + waits for readiness.
    // bumps in_flight_ for the lifetime of the returned response.
    server_http_res_ptr proxy(const server_http_req & req, const std::string & method);

    // Same as proxy() but does NOT bump in_flight_ AND never lazily spawns the
    // child (use for non-chat metadata endpoints like /v1/models / /props).
    // Returns 503 when the child is down so a routine poll can't re-fork an
    // intentionally-evicted worker back into VRAM.
    server_http_res_ptr proxy_noflight(const server_http_req & req, const std::string & method);

    // Child port (random ephemeral). 0 before first ensure_loaded.
    int child_port() const { return child_port_; }
    int parent_port() const { return parent_port_; }

private:
    // Forward to the already-running child; never spawns. Returns 503 if down.
    server_http_res_ptr forward_to_child(const server_http_req & req, const std::string & method);
    bool spawn_locked();  // caller holds spawn_mutex_
    bool wait_for_child_health_locked(int timeout_ms);
    void reap_locked();   // caller holds spawn_mutex_

    // captured at construction
    std::string argv0_;
    std::vector<std::string> child_argv_;
    int parent_port_ = 0;
    int read_timeout_ = 600;
    int write_timeout_ = 600;

    // serialises spawn/shutdown — proxy itself is wait-free once child
    // is alive (we re-check is_loaded_ inside ensure_loaded fast path).
    mutable std::mutex spawn_mutex_;
    std::shared_ptr<subprocess_s> subproc_;
    // background log-relay thread (drains child stdout+stderr → parent
    // stderr) so the child's log pipe never fills + blocks the child.
    std::thread log_thread_;
    std::atomic<int> child_port_{0};
    std::atomic<bool> loaded_{false};
    std::atomic<bool> draining_{false};
    std::atomic<int>  in_flight_{0};
};
