// Minimal per-op CUDA profiler — env-gated `GGML_CUDA_PROFILE_OPS=1`.
//
// When enabled, wraps each `ggml_cuda_compute_forward` call with a cudaEvent
// pair, synchronizes after every op, and accumulates (total_us, count) keyed
// both by op-type and by individual node name. Dumps a top-N table every
// N graph_compute calls (default 200, override via GGML_CUDA_PROFILE_EVERY).
//
// Cost: ~1-3% wall-clock overhead (a sync per op + map lookups). Forces
// `use_cuda_graph=false` for the captured eval so per-op timing is real,
// not amortized across a graph replay.
//
// Used only when LLAMA_DEV_PROFILE / GGML_CUDA_DEV_PROFILE build flag is ON.

#pragma once

#include "common.cuh"
#include "ggml.h"
#include <string>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#ifdef GGML_CUDA_DEV_PROFILE

namespace ggml_cuda_op_profile {

struct entry {
    double total_us = 0.0;
    uint64_t count  = 0;
};

struct state {
    std::mutex mu;
    std::unordered_map<std::string, entry> by_op;
    std::unordered_map<std::string, entry> by_node;
    uint64_t graph_calls = 0;
    uint64_t dump_every  = 200;
    bool enabled         = false;
    bool initialized     = false;

    void maybe_init() {
        if (initialized) return;
        initialized = true;
        const char * env = getenv("GGML_CUDA_PROFILE_OPS");
        enabled = (env && env[0] == '1');
        const char * env_n = getenv("GGML_CUDA_PROFILE_EVERY");
        if (env_n) {
            dump_every = std::max(1, atoi(env_n));
        }
    }
};

inline state & get_state() {
    static state s;
    s.maybe_init();
    return s;
}

inline bool enabled() {
    return get_state().enabled;
}

inline void add(const char * op_name, const char * node_name, double us) {
    auto & s = get_state();
    std::lock_guard<std::mutex> lk(s.mu);
    auto & e1 = s.by_op[op_name];
    e1.total_us += us; e1.count++;
    auto & e2 = s.by_node[node_name];
    e2.total_us += us; e2.count++;
}

inline void dump(FILE * out) {
    auto & s = get_state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.by_op.empty()) return;

    fprintf(out, "\n========== ggml-cuda profile (graph_calls=%llu) ==========\n",
            (unsigned long long)s.graph_calls);

    {
        std::vector<std::pair<std::string, entry>> v(s.by_op.begin(), s.by_op.end());
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) {
            return a.second.total_us > b.second.total_us;
        });
        double tot = 0;
        for (auto & p : v) tot += p.second.total_us;
        fprintf(out, "-- per op-type --  (total %.1f ms across %llu graph_calls)\n",
                tot/1000.0, (unsigned long long)s.graph_calls);
        fprintf(out, "%-22s  %10s  %10s  %10s  %8s\n",
                "op", "total_ms", "count", "us/call", "pct");
        for (auto & p : v) {
            double tot_ms = p.second.total_us / 1000.0;
            double us_per = p.second.total_us / p.second.count;
            double pct    = 100.0 * p.second.total_us / tot;
            fprintf(out, "%-22s  %10.2f  %10llu  %10.2f  %7.2f%%\n",
                    p.first.c_str(), tot_ms,
                    (unsigned long long)p.second.count, us_per, pct);
        }
    }

    {
        std::vector<std::pair<std::string, entry>> v(s.by_node.begin(), s.by_node.end());
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) {
            return a.second.total_us > b.second.total_us;
        });
        const size_t N = std::min((size_t)25, v.size());
        fprintf(out, "-- top %zu individual nodes (by total_us) --\n", N);
        fprintf(out, "%-50s  %10s  %10s  %10s\n", "node", "total_ms", "count", "us/call");
        for (size_t i = 0; i < N; i++) {
            const auto & p = v[i];
            double tot_ms = p.second.total_us / 1000.0;
            double us_per = p.second.total_us / p.second.count;
            fprintf(out, "%-50s  %10.2f  %10llu  %10.2f\n",
                    p.first.c_str(), tot_ms,
                    (unsigned long long)p.second.count, us_per);
        }
    }
    fprintf(out, "==========================================================\n\n");
    fflush(out);
}

inline void note_graph_call() {
    auto & s = get_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.graph_calls++;
    if (s.dump_every > 0 && (s.graph_calls % s.dump_every) == 0) {
        // unlock before dump (dump re-locks)
    }
}

// Internal RAII helper — record event before op, query elapsed after.
struct event_scope {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    cudaStream_t stream;
    const char * op_name;
    const char * node_name;
    bool ok = false;

    event_scope(cudaStream_t s, const char * op, const char * node)
        : stream(s), op_name(op), node_name(node) {
        if (!enabled()) return;
        if (cudaEventCreate(&e0) != cudaSuccess) return;
        if (cudaEventCreate(&e1) != cudaSuccess) { cudaEventDestroy(e0); e0 = nullptr; return; }
        cudaEventRecord(e0, stream);
        ok = true;
    }

    ~event_scope() {
        if (!ok) return;
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e0, e1);
        add(op_name, node_name, ms * 1000.0);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
};

} // namespace ggml_cuda_op_profile

#define GGML_CUDA_PROFILE_OP_SCOPE(stream, op_name, node_name) \
    ::ggml_cuda_op_profile::event_scope _ggml_cuda_op_prof_scope((stream), (op_name), (node_name))

#define GGML_CUDA_PROFILE_NOTE_GRAPH_CALL() \
    do { if (::ggml_cuda_op_profile::enabled()) ::ggml_cuda_op_profile::note_graph_call(); } while (0)

#define GGML_CUDA_PROFILE_DUMP_IF_DUE(out) \
    do { \
        auto & _s = ::ggml_cuda_op_profile::get_state(); \
        if (_s.enabled && _s.dump_every > 0 && (_s.graph_calls % _s.dump_every) == 0 && _s.graph_calls > 0) { \
            ::ggml_cuda_op_profile::dump(out); \
        } \
    } while (0)

#define GGML_CUDA_PROFILE_FORCE_NO_CUDA_GRAPH() \
    (::ggml_cuda_op_profile::enabled())

#else  // !GGML_CUDA_DEV_PROFILE

#define GGML_CUDA_PROFILE_OP_SCOPE(stream, op_name, node_name) do {} while (0)
#define GGML_CUDA_PROFILE_NOTE_GRAPH_CALL()                    do {} while (0)
#define GGML_CUDA_PROFILE_DUMP_IF_DUE(out)                     do {} while (0)
#define GGML_CUDA_PROFILE_FORCE_NO_CUDA_GRAPH()                (false)

#endif  // GGML_CUDA_DEV_PROFILE
