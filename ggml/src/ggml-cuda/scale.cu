#include "scale.cuh"

#include <cuda_fp16.h>

#define MAX_GRIDDIM_X 0x7FFFFFFF

// Scale storage may be F32 or F16. Compute is always F32; only the global load
// and store are templated on the storage dtype. F16 is needed for the GDN
// recurrent-state buffer (LLAMA_GDN_STATE_F16) — build_rs zero-clears state
// slots via ggml_scale_inplace(view, 0) and that view is then F16.
template <typename T>
static __device__ __forceinline__ float scale_load(const T * p) { return static_cast<float>(*p); }
template <>
__device__ __forceinline__ float scale_load<half>(const half * p) { return __half2float(*p); }

template <typename T>
static __device__ __forceinline__ void scale_store(T * p, float v) { *p = static_cast<T>(v); }
template <>
__device__ __forceinline__ void scale_store<half>(half * p, float v) { *p = __float2half(v); }

template <typename T>
static __global__ void scale_kern(const T * x, T * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        scale_store<T>(dst + i, scale * scale_load<T>(x + i) + bias);
    }
}

template <typename T>
static void scale_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_kern<T>, launch_params, x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        scale_cuda<half>((const half *) src0->data, (half *) dst->data, scale, bias, ggml_nelements(src0), stream);
    } else {
        scale_cuda<float>((const float *) src0->data, (float *) dst->data, scale, bias, ggml_nelements(src0), stream);
    }
}
