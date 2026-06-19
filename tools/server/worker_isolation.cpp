// worker_isolation.cpp — see worker_isolation.h.
//
// Implementation notes:
//   - Child spawn uses sheredom/subprocess (same as the router server)
//     so we get cross-platform process handling out of the box.
//   - We do NOT capture stdout — the child logs directly to stderr and
//     we want those log lines visible in `docker logs` without an extra
//     marshalling thread. SUBPROCESS_OPTION_INHERIT_ENVIRONMENT preserves
//     env across exec.
//   - Health probe: parent polls http://127.0.0.1:<child_port>/health
//     every 250 ms until 200, up to 60 s (warm-page-cache 5-10s, cold
//     ~20-30s). Same shape as docker-compose's healthcheck loop.

#include "worker_isolation.h"
#include "server-common.h"
#include "server-models.h"      // for server_http_proxy
#include "common.h"

#include <cpp-httplib/httplib.h>
#include <sheredom/subprocess.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#ifdef _WIN32
#  include <winsock2.h>
#  include <windows.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
extern char ** environ;
#endif

namespace {

// Same helper as server-models.cpp; kept private here so we don't
// expose more of that translation unit's internals.
int get_free_port() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return -1;
    typedef SOCKET sock_t;
#  define INV INVALID_SOCKET
#  define CLOSEME(s) closesocket(s)
#else
    typedef int sock_t;
#  define INV -1
#  define CLOSEME(s) ::close(s)
#endif
    sock_t s = ::socket(AF_INET, SOCK_STREAM, 0);
    if (s == INV) {
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(0);
    if (::bind(s, (sockaddr *) &addr, sizeof(addr)) != 0) {
        CLOSEME(s);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }
#ifdef _WIN32
    int len = sizeof(addr);
#else
    socklen_t len = sizeof(addr);
#endif
    if (::getsockname(s, (sockaddr *) &addr, &len) != 0) {
        CLOSEME(s);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }
    int port = ntohs(addr.sin_port);
    CLOSEME(s);
#ifdef _WIN32
    WSACleanup();
#endif
    return port;
}

// Read /proc/self/exe (Linux) or fall back to argv[0]. The subprocess
// API wants a full path so it can execv directly without /bin/sh.
std::string self_exe_path(const char * argv0_fallback) {
#ifdef __linux__
    char buf[4096];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::string(buf);
    }
#endif
    return std::string(argv0_fallback ? argv0_fallback : "llama-server");
}

// Build child argv by stripping out --host/--port (we override) + the
// usual reserved-arg pattern. We pass the child the same model/mmproj/
// quantization/sampling/etc. flags so its config is bit-identical to
// the user's intent — only HOST + PORT differ.
//
// `argv` here is the parent's argv. We skip:
//   * `--host` / `-H` / `--hostname` <val>  — we force 127.0.0.1
//   * `--port` / `-p`                <val>  — we force the random port
// Everything else flows through.
std::vector<std::string> sanitize_child_argv(int argc, char ** argv) {
    std::vector<std::string> out;
    out.reserve(argc);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--host" || a == "-H" || a == "--hostname" ||
            a == "--port" || a == "-p") {
            if (i + 1 < argc) i++;  // skip its value
            continue;
        }
        out.push_back(std::move(a));
    }
    return out;
}

} // namespace

worker_isolation::worker_isolation(const common_params & params, int argc, char ** argv) {
    argv0_         = self_exe_path(argv[0]);
    child_argv_    = sanitize_child_argv(argc, argv);
    parent_port_   = params.port;
    read_timeout_  = params.timeout_read  > 0 ? params.timeout_read  : 600;
    write_timeout_ = params.timeout_write > 0 ? params.timeout_write : 600;
}

worker_isolation::~worker_isolation() {
    // Kill the child on parent exit so we don't leak a process.
    std::lock_guard<std::mutex> lk(spawn_mutex_);
    reap_locked();
}

bool worker_isolation::is_loaded() const {
    return loaded_.load() && subproc_ != nullptr;
}

bool worker_isolation::ensure_loaded(const std::string & want_gpu) {
    // Resolve the target card: explicit override, else the configured default.
    // Empty → inherit the container CUDA_VISIBLE_DEVICES (legacy behaviour).
    const std::string target = want_gpu.empty() ? default_gpu_ : want_gpu;

    // Fast path: live child already on the right card.
    if (loaded_.load() && subproc_ && subprocess_alive(subproc_.get()) &&
        worker_gpu_ == target) {
        return true;
    }
    std::lock_guard<std::mutex> lk(spawn_mutex_);
    if (loaded_.load() && subproc_ && subprocess_alive(subproc_.get()) &&
        worker_gpu_ == target) {
        return true;
    }
    // Relocation: a live child on a different card must be killed + respawned
    // on the target. Drain first so in-flight proxy streams aren't severed
    // mid-response (handler-entry drain gate already 503's NEW requests; the
    // request driving this relocation is itself pre-drain by contract — the
    // gate sequences drain→wait→relocate upstream).
    if (subproc_ && subprocess_alive(subproc_.get()) && worker_gpu_ != target) {
        SRV_INF("worker-isolation: relocating child GPU '%s' -> '%s'\n",
                worker_gpu_.empty() ? "(inherit)" : worker_gpu_.c_str(),
                target.empty()      ? "(inherit)" : target.c_str());
        reap_locked();
    } else if (subproc_ && !subprocess_alive(subproc_.get())) {
        // Child existed but died; reap before respawn.
        reap_locked();
    }
    want_gpu_ = target;
    return spawn_locked();
}

bool worker_isolation::shutdown_child() {
    std::lock_guard<std::mutex> lk(spawn_mutex_);
    if (!subproc_) return false;
    bool alive = subprocess_alive(subproc_.get());
    reap_locked();
    return alive;
}

void worker_isolation::reap_locked() {
    if (!subproc_) return;
    if (subprocess_alive(subproc_.get())) {
        // SIGKILL on POSIX, TerminateProcess on Windows — same effect:
        // primary CUDA context teardown is immediate, all VRAM is
        // reclaimed by the kernel.
        subprocess_terminate(subproc_.get());
    }
    int code = 0;
    subprocess_join(subproc_.get(), &code);
    subprocess_destroy(subproc_.get());
    subproc_.reset();
    child_port_.store(0);
    loaded_.store(false);
    worker_gpu_.clear();
    // log_thread_ exits when fgets hits EOF, which subprocess_destroy
    // triggers by closing the read fd. Join here so the next spawn
    // doesn't race.
    if (log_thread_.joinable()) log_thread_.join();
}

bool worker_isolation::spawn_locked() {
    int port = get_free_port();
    if (port <= 0) {
        SRV_ERR("%s", "worker-isolation: failed to allocate child port\n");
        return false;
    }

    // Compose argv: <exe> <user-args-minus-host-port> --host 127.0.0.1 --port <child_port>
    std::vector<std::string> args;
    args.reserve(child_argv_.size() + 5);
    args.push_back(argv0_);
    for (const auto & a : child_argv_) args.push_back(a);
    args.emplace_back("--host");
    args.emplace_back("127.0.0.1");
    args.emplace_back("--port");
    args.emplace_back(std::to_string(port));

    std::vector<char *> argv_p;
    argv_p.reserve(args.size() + 1);
    for (auto & s : args) argv_p.push_back(const_cast<char *>(s.c_str()));
    argv_p.push_back(nullptr);

    // env: inherit + force LLAMA_WORKER_ISOLATION=0 so the child
    // doesn't re-recurse, + LLAMA_WORKER_ISOLATION_CHILD=1 as a
    // sentinel future-readers can grep for.
    std::vector<std::string> env_owned;
#ifdef _WIN32
    // not implementing the Windows env-block path here; subprocess.h's
    // inherit-environment option suffices on Windows.
#else
    // When pinning to a specific card we override CUDA_VISIBLE_DEVICES in the
    // child env (parent is CUDA-free, so its primary context lands on the
    // requested device). Empty want_gpu_ → leave the inherited value untouched
    // (fully backward compatible).
    const bool pin_gpu = !want_gpu_.empty();
    if (environ) {
        for (char ** e = environ; *e; ++e) {
            std::string ev = *e;
            // Drop any prior LLAMA_WORKER_ISOLATION* — we set our own.
            if (ev.rfind("LLAMA_WORKER_ISOLATION", 0) == 0) continue;
            // Drop inherited CUDA_VISIBLE_DEVICES when we're pinning — we
            // re-add the targeted value below.
            if (pin_gpu && ev.rfind("CUDA_VISIBLE_DEVICES=", 0) == 0) continue;
            env_owned.push_back(std::move(ev));
        }
    }
#endif
    env_owned.emplace_back("LLAMA_WORKER_ISOLATION=0");
    env_owned.emplace_back("LLAMA_WORKER_ISOLATION_CHILD=1");
    if (pin_gpu) {
        env_owned.emplace_back("CUDA_VISIBLE_DEVICES=" + want_gpu_);
    }

    std::vector<char *> envp;
    envp.reserve(env_owned.size() + 1);
    for (auto & s : env_owned) envp.push_back(const_cast<char *>(s.c_str()));
    envp.push_back(nullptr);

    SRV_INF("worker-isolation: spawning child on 127.0.0.1:%d\n", port);
    for (size_t i = 0; i < args.size(); ++i) {
        SRV_INF("  argv[%zu] = %s\n", i, args[i].c_str());
    }

    subproc_ = std::make_shared<subprocess_s>();
    // Combine the child's stdout + stderr into one pipe so we can drain
    // it with a single reader thread; without a drain, the child's log
    // pipe fills + blocks the child's writes during model load (was
    // costing us 2-minute spawn-time hangs on cold reload paths).
    int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
    int rc = subprocess_create_ex(argv_p.data(), options, envp.data(), subproc_.get());
    if (rc != 0) {
        SRV_ERR("%s", "worker-isolation: subprocess_create_ex failed\n");
        subproc_.reset();
        return false;
    }

    child_port_.store(port);

    // Start the log-relay thread now. It exits when the child's pipe
    // hits EOF (subprocess teardown closes the read end).
    if (log_thread_.joinable()) log_thread_.join();
    {
        auto sp = subproc_;
        log_thread_ = std::thread([sp, port]() {
            FILE * out = subprocess_stdout(sp.get());
            if (!out) return;
            char buf[8 * 1024];
            while (fgets(buf, sizeof(buf), out) != nullptr) {
                // Forward verbatim with a per-port prefix so concurrent
                // child lines are distinguishable from parent lines.
                std::fprintf(stderr, "[child:%d] %s", port, buf);
            }
        });
    }

    // Poll child health until ready or timeout.
    constexpr int health_timeout_ms = 120 * 1000; // 2 min cold-load
    if (!wait_for_child_health_locked(health_timeout_ms)) {
        SRV_ERR("worker-isolation: child failed to become healthy within %d ms — killing\n",
                health_timeout_ms);
        reap_locked();
        return false;
    }

    loaded_.store(true);
    worker_gpu_ = want_gpu_;
    SRV_INF("worker-isolation: child ready on 127.0.0.1:%d (pid alive) gpu=%s\n",
            port, worker_gpu_.empty() ? "(inherit)" : worker_gpu_.c_str());
    return true;
}

bool worker_isolation::wait_for_child_health_locked(int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    httplib::Client cli("127.0.0.1", child_port_.load());
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(2, 0);
    while (std::chrono::steady_clock::now() < deadline) {
        // If the child died, give up immediately.
        if (!subprocess_alive(subproc_.get())) {
            SRV_ERR("%s", "worker-isolation: child died during startup\n");
            return false;
        }
        auto res = cli.Get("/health");
        if (res && res->status == 200) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    return false;
}

// Internal helper: builds a server_http_proxy wrapped to release
// in_flight_ on destruction (matches lifetime of the streaming response).
namespace {

struct proxied_with_in_flight : server_http_res {
    // Holds the underlying proxy + the in-flight guard. The proxy is
    // a server_http_res itself (subclass), so we wrap its `next` and
    // copy out status/headers/content_type/data.
    std::unique_ptr<server_http_res>           inner;
    std::unique_ptr<worker_isolation::in_flight_guard> ifg;
    proxied_with_in_flight(std::unique_ptr<server_http_res> p,
                           std::unique_ptr<worker_isolation::in_flight_guard> g)
        : inner(std::move(p)), ifg(std::move(g)) {
        status       = inner->status;
        headers      = inner->headers;
        content_type = inner->content_type;
        data         = inner->data;
        if (inner->is_stream()) {
            // bind a shared_ptr so the wrapper survives until last chunk.
            auto inner_raw = inner.get();
            this->next = [inner_raw](std::string & out) -> bool {
                return inner_raw->next(out);
            };
        }
    }
};

} // namespace

// Forward to the already-running child. NEVER spawns — callers that want a
// lazy cold-load must call ensure_loaded() first. Returns 503 when the child
// is down so a metadata poll can't drag the model back into VRAM.
server_http_res_ptr worker_isolation::forward_to_child(const server_http_req & req, const std::string & method) {
    int port = child_port_.load();
    if (port <= 0 || !loaded_.load()) {
        auto res = std::make_unique<server_http_res>();
        res->status = 503;
        res->data = safe_json_to_str({{"error", format_error_response("worker child not loaded", ERROR_TYPE_UNAVAILABLE)}});
        return res;
    }
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) proxy_path += '?' + req.query_string;
    auto proxy = std::make_unique<server_http_proxy>(
        method,
        "http",
        "127.0.0.1",
        port,
        proxy_path,
        req.headers,
        req.body,
        req.files,
        req.should_stop,
        read_timeout_,
        write_timeout_);
    return proxy;
}

server_http_res_ptr worker_isolation::proxy_noflight(const server_http_req & req, const std::string & method) {
    // Metadata / introspection endpoints (/v1/models, /props, /metrics,
    // /slots, tokenize, …). These must NOT lazily spawn the child: an
    // idle/evicted worker (e.g. while an Avatar/Flux render holds the GPU and
    // kob-gpu-gate has unloaded the LLM) would be re-forked by a routine
    // health/model-name poll, double-loading the model into VRAM on top of
    // the render → OOM. Forward only when the child is already up; else 503.
    return forward_to_child(req, method);
}

std::string worker_isolation::resolve_target_gpu(const server_http_req & req) const {
    // Case-insensitive lookup of "X-Worker-Gpu". httplib preserves the
    // client's header casing, so we can't rely on a fixed key. Empty value
    // (or absent header) → keep current/default card.
    for (const auto & [k, v] : req.headers) {
        if (k.size() != std::string("X-Worker-Gpu").size()) continue;
        bool match = true;
        const char * want = "x-worker-gpu";
        for (size_t i = 0; i < k.size(); ++i) {
            if (std::tolower((unsigned char) k[i]) != want[i]) { match = false; break; }
        }
        if (match) {
            // trim surrounding whitespace
            size_t a = v.find_first_not_of(" \t");
            if (a == std::string::npos) return std::string();
            size_t b = v.find_last_not_of(" \t");
            return v.substr(a, b - a + 1);
        }
    }
    return std::string();
}

server_http_res_ptr worker_isolation::proxy(const server_http_req & req, const std::string & method) {
    // Inference path: this is the only place allowed to lazily cold-load the
    // child (the keep-warm "reload on next chat" contract). kob-gpu-gate gates
    // these upstream so a chat can't reload the model mid-render.
    //
    // X-Worker-Gpu header (per-request override) places / relocates the child
    // onto a specific card before serving. Empty/absent → default_gpu_.
    const std::string target_gpu = resolve_target_gpu(req);
    if (!ensure_loaded(target_gpu)) {
        auto res = std::make_unique<server_http_res>();
        res->status = 503;
        res->data = safe_json_to_str({{"error", format_error_response("worker child failed to start", ERROR_TYPE_UNAVAILABLE)}});
        return res;
    }
    auto raw = forward_to_child(req, method);
    if (raw->status >= 500 && raw->data.find("worker child") != std::string::npos) {
        // Don't wrap an error response with an in-flight guard.
        return raw;
    }
    auto guard = std::make_unique<in_flight_guard>(this);
    return std::make_unique<proxied_with_in_flight>(std::move(raw), std::move(guard));
}
