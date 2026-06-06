# Handoff — turboquant upstream sync + gemma4 + PR #23398 + dbrain/ggml (2026-06-06)

Goal: get llama.cpp-turboquant current enough to run **gemma4-12b** (which needs newer
llama.cpp). Done overnight via plain git. **No GPU was used** (an active `songgen-generate`
bench was running on the RTX 3060 throughout; compiling uses 0 VRAM).

**COMPILE-VERIFIED** (host: pixi cmake 4.3.3 + gcc-16 for CPU, pixi nvcc 13.3 + g++-14 for CUDA):
- Both branches: full **CPU** `llama-server` build clean — **0 errors** (34 pre-existing
  upstream warnings, none in merged files). dbrain branch's `GGML_OP_COUNT==99` static_asserts
  pass.
- Every conflict-touched **`.cu`** file compiles under `nvcc -arch=sm_86` (per-file `-c`):
  `fattn-common.cuh` (both branches), and on dbrain: `snake.cu` (the union), `concat.cu`,
  `im2col.cu`, `conv-1d-direct.cu`, `ggml-cuda.cu`, `getrows.cu`, `conv-transpose-1d.cu`,
  `acc.cu`, `rope.cu`, `fattn.cu`.
- Caveat: toolchain was **CUDA 13.3 / g++-14**, prod is **CUDA 12.9 / gcc-13**; and these were
  per-file compiles, not a full `-DGGML_CUDA=ON` link. Standard code, low risk — but still do
  the real koblem `make docker-push-llama` (CUDA 12.9) build before deploying.

## TL;DR — what to build

- **Deliverable branch (the goal): `feature/upstream-sync-2026-06-06-gemma4`**
  Steps 1–3. Upstream-synced + gemma4 + gemma4-mtp. The in-tree ggml is the clean
  upstream+turboquant ggml (untouched by step 4). Build/test THIS for gemma4-12b.
- **Optional branch: `feature/ggml-dbrain-kernels-2026-06-06`**
  Step 4 only (the dbrain/ggml media kernels), branched off the deliverable. Kept
  separate on purpose so a ggml mistake can't break the gemma4 build. CUDA-only media
  kernels, irrelevant to gemma4/LLM inference. Build-test separately if you actually want
  a unified in-tree ggml.
- **Safety backup tag: `backup-pre-gemma4-sync-2026-06-06`** = the pre-sync prod tip
  (`102a8aadc`, feature/turboquant-kv-cache).

Nothing pushed. Nothing built. Branches are local.

## Branch topology

```
backup-pre-gemma4-sync-2026-06-06  (102a8aadc)  <- prod tip before tonight
        │
feature/upstream-sync-2026-06-06-gemma4
  102a8aadc (turboquant-kv-cache origin tip)
   └─ bfb6731bc  Merge upstream/master (140 commits: 19e92c33e..6b80c74f2)  [step 2]
       └─ 373ed6ebb  Merge gemma4-mtp PR #23398 (am17an:gemma4-mtp)         [step 3]   <-- DELIVERABLE TIP
            └─ feature/ggml-dbrain-kernels-2026-06-06
                 66c42808f  ggml: merge dbrain/ggml 25 commits (BUILD-UNVERIFIED)   [step 4]
```

## Step 1 — origin sync
`feature/turboquant-kv-cache` was already up to date with `origin/feature/turboquant-kv-cache`
(`102a8aadc`). The deliverable branch is cut from there. (Prod ref in memory was
`upstream-sync-2026-05-29 @ 7f86ab1e`; `102a8aadc` is two worker-isolation commits ahead of it.)

## Step 2 — merge ggml-org/llama.cpp master (140 commits, 19e92c33e..6b80c74f2)
Brings gemma4 + many archs, ggml updates, server refactors. Merge commit `bfb6731bc`.
8 conflicts, resolved:

| File | Resolution |
|---|---|
| `.devops/nix/package.nix` | took upstream webui deps (`nodejs`, `importNpmLock`); kept our `spirv-headers` dedup |
| `README.md` | kept turboquant fork badges/description |
| `conversion/__init__.py` | union: our `Marlin` mapping + upstream's `Mellum` |
| `src/llama-context.cpp` | kept turbo head-128-padding in the KV blck-size validation loops; adopted upstream's `n_layer()` bound |
| `src/llama-kv-cache.cpp` | see below (the big one) |
| `tools/server/server-context.cpp` | kept our deletion of the `--fit` draft/MTP VRAM-reservation + the MTP `type_k/type_v` overrides; adopted upstream's `cparams_mtp.n_outputs_max = n_parallel` |
| `tools/server/server-http.cpp` | kept our embedded `LLAMA_BUILD_UI` bundle serving (not upstream's asset-registry path) |
| `ggml/src/ggml-cuda/fattn-common.cuh` | adopted upstream's `f16_extra` dequant-scratch refactor; **dropped** the obsolete turboquant HIP pool-bypass OOM fix (see risk note) |

### The load-bearing upstream API change: `hparams.n_layer`
Upstream **renamed the field `n_layer` → `n_layer_all`** and added a **method `n_layer()`**
(effective layers = excludes the `n_layer_nextn` MTP layers). It also **removed
`n_layer_kv()`**. `n_layer_all == old n_layer` exactly; `n_layer_all = n_layer() + n_layer_nextn`.

This silently breaks any turboquant `hparams.n_layer` field access that auto-merged. Swept the
whole tree; the real `llama_hparams` uses (false positives in `common/fit.cpp` and
`tools/mtmd/*` use their own structs) were fixed:
- `src/llama-context.cpp` validation loops → `n_layer()` (matches upstream's intent for those loops).
- `src/llama-kv-cache.cpp`: upstream renamed the ctor local `n_layer_kv` (=`hparams.n_layer_kv()`)
  to `n_layer` (=`hparams.n_layer_all`) and now skips non-KV layers via `has_kv(il)`. Kept our
  auto-asymmetric GQA-ratio K-upgrade block, the `+3` turbo-rotation tensor-overhead, and the
  layer-adaptive Boundary-V policy — rebased onto the new `n_layer` local and dropped the
  now-shadowing inner `const uint32_t n_layer = hparams.n_layer;` redecl. Fixed the two
  auto-merged `hparams.n_layer` field accesses (lines ~293/301).
- `src/models/qwen35.cpp:494` already correctly uses `n_layer()` (MTP block lives at index
  `n_layer()`); only its stale comment was corrected.
- `attn_rot`: kept turboquant's OFF-by-default env-gated design; carried ONLY upstream's
  DeepSeek-V3.2 DSA forced-rotation special-case (no-op for archs we run; symbols verified present).

**⚠️ Build-machine check:** anywhere the diff between `n_layer()` and `n_layer_all` matters is
MTP/nextn models (Qwen3.5, gemma4-mtp). I preserved pre-refactor behavior (`n_layer_all`) for
sizing/over-allocation and adopted `n_layer()` where upstream did for per-attention-layer loops.
Sanity-check the KV-cache adaptive boundary policy on a Qwen3.5 MTP run if you use
`TURBO_LAYER_ADAPTIVE`/Boundary-V (the "last N layers" boundary now counts against `n_layer_all`).

### fattn HIP risk note
Upstream replaced the pool-allocated `K_f16`/`V_f16` dequant scratch with pre-reserved buffers
(`f16_extra.K/.V`). The turboquant HIP/ROCm pool-bypass OOM fix operated on the old pool alloc,
which no longer exists, so it was dropped. **Prod runs CUDA on the RTX 3060 — the HIP path is not
exercised.** If quantized-KV OOM ever resurfaces on HIP/ROCm, re-apply the fix inside the new
`ggml_cuda_flash_attn_ext_get_f16_extra_data` allocation path, not in `fattn-common.cuh`.

## Step 3 — PR #23398 (am17an:gemma4-mtp), merge commit `373ed6ebb`
12 commits / 21 files: gemma4-assistant draft arch (`gemma4-assistant`), MTP graph types,
conversion + gguf tensor maps, `Gemma4Model` assistant variants, `src/models/gemma4-assistant.cpp`.
The PR's merge-base was already inside our step-2 merge, so the 3-way isolated just the feature delta.

1 conflict — `tools/server/server-context.cpp`: the PR restored the `--fit` draft/MTP
VRAM-reservation block (plus a gemma4-assistant `skip_measure` temp-hack). **Kept it removed**
(turboquant deleted that block; MTP is VRAM-free; draft sizing is manual here). With the block
gone there is nothing for the PR's skip-hack to guard. Our `cparams_mtp` setup (no `type_k/type_v`
override, `n_outputs_max`, MTP `n_ubatch` cap) is preserved; PR gemma4 logic auto-merged around it.

Notes:
- PR is **upstream WIP** (carries its own "temp hack / rm later" markers). gemma4 MTP uses a
  **separate assistant draft model** (`has_draft` path), unlike Qwen3.5's in-weights MTP.
- The PR says MTP works for the 31B and 26B-4B variants, **not** E4B/E2B. For plain
  **gemma4-12b** you don't need MTP at all — the base `LLM_ARCH_GEMMA4` inference path (from the
  step-2 upstream merge) is what runs it; MTP is an optional speed-up if a matching assistant
  draft model exists.

### gemma4 support — verified present (git, not build)
`LLM_ARCH_GEMMA4` (`src/llama-arch.{h,cpp}`), `src/models/gemma4.cpp` + `gemma4-assistant.cpp`,
`conversion/gemma.py` `Gemma4Model`/`Gemma4UnifiedModel` (registered for
`Gemma4ForConditionalGeneration`/`Gemma4ForCausalLM`/`Gemma4Unified*`), gguf constants + tensor maps.

## Step 4 — dbrain/ggml 25 commits → in-tree ggml (SEPARATE branch, BUILD-UNVERIFIED)
`feature/ggml-dbrain-kernels-2026-06-06`, commit `66c42808f`.

**Why separate:** these are CUDA **media kernels** (TTS/avatar/music): snake op, conv-1d-direct,
vocoder conv-transpose F16 wmma, graph/op/mul_mat hooks, Q4_K get_rows, concat/acc/rope F16+I32,
stream-priority. They have **no consumer in the LLM build** (nothing here calls `ggml_snake()` etc.)
and are irrelevant to gemma4-12b. A botched ggml merge breaks the *entire* build (incl. gemma4),
so it's isolated. Applied as a cross-repo 3-way (`git apply --3way --directory=ggml`) of
`ac6f7b44..dbrain/master`, path-remapped root→`ggml/`.

**Major finding:** upstream has since **absorbed overlapping functionality**, so this is not a clean
add — several dbrain commits now compete with upstream:
- Upstream `#22667` added a CUDA **snake fusion** (`ggml_cuda_op_snake_fused`, no `GGML_OP`).
  Kept BOTH it and dbrain's standalone `GGML_OP_SNAKE` op; renamed dbrain's kernel
  `snake_kernel`→`snake_op_kernel` to avoid collision. (Different math: fusion gets post-exp
  `a/inv_b`; standalone gets raw `alpha/beta` and exps in-kernel.)
- `im2col.cu`: upstream (in-kernel grid-stride over `iow`) and dbrain (host-side `iow_base`
  chunking) both fixed `OW>65535`. Took **upstream's**; removed dbrain's now-unused `iow_base`
  kernel param. Functionally equivalent, nothing lost.
- `concat.cu`: dbrain templated `concat_f32_cont`→`concat_T_cont<T>` (F16/I32). Took dbrain's
  templated raw launch for `dim==0` to match `dim==1/2`; dropped upstream's
  `ggml_cuda_kernel_launch` wrapper (only touched the old F32-only path).
- `dequantize.cuh`, `ggml-cuda.cu` get_rows switch, `ggml-cuda.cu` hooks block: clean unions.

**GGML_OP_COUNT bookkeeping:** base 96, upstream +1 (=97), dbrain +2 (SNAKE, CONV_1D_DIRECT) (=98).
Combined = **99**. Verified `GGML_OP_NAME[]` and `GGML_OP_SYMBOL[]` each have exactly 99 entries;
set all 3 `static_assert`s (`ggml.c` ×2, `ggml-rpc.h`) to 99. Both new ops verified fully wired
across enum / API / builders / CPU dispatch+forward / CUDA dispatch+supports_op.

**Compile-verified (2026-06-06):** CPU build clean (0 errors; the `GGML_OP_COUNT==99`
static_assert triple passes); the unioned `snake.cu`, `conv-1d-direct.cu`, F16-wmma
`conv-transpose-1d.cu`, and `ggml-cuda.cu` (hooks + get_rows union) all compile under
`nvcc -arch=sm_86`. `-Wswitch` is moot anyway — prod builds with `-Wno-error`. Residual: full
`-DGGML_CUDA=ON` link + prod CUDA 12.9 toolchain not exercised here (used 13.3); do the real
Docker build before relying on this branch.

**Decision to confirm:** do you actually want the media kernels in the LLM's in-tree ggml at all?
Turboquant deliberately kept its own in-tree ggml separate from the consolidated `dbrain/ggml`.
If the answer is "no / not needed for gemma4," just ignore this branch — the deliverable branch
doesn't include it.

## Conflict-resolution memory aids
`git rerere` was enabled for this work, so re-doing any of these merges will auto-replay the
resolutions. The cross-repo dbrain patches are at `/tmp/dbrain-ggml-main.patch` and
`/tmp/dbrain-ggml-tests.patch` (regenerate from `~/dev/ggml`: `git diff --full-index
ac6f7b44 master`).

## Recommended next steps (build machine, GPU)
1. Build the **deliverable branch** via the prod path (koblem `make docker-push-llama`, CUDA 12.9).
   CPU + per-file nvcc compiles already pass here; this is the full CUDA-12.9 link confirmation.
2. Convert + run **gemma4-12b** (no MTP needed). Confirm `LLM_ARCH_GEMMA4` loads and decodes.
   (This is the first actual GPU run — deferred per your "no GPU yet".)
3. (Optional) regression-test Qwen3.5-VL prod config to confirm the upstream merge didn't move
   turboquant KV/MTP behavior — especially if you use `TURBO_LAYER_ADAPTIVE`.
4. (Optional) build-test `feature/ggml-dbrain-kernels-2026-06-06` only if you want the unified ggml.
5. Push whichever branches you keep; bump the koblem/kobbler llama ref if deploying.
