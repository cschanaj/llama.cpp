#include "weightedsum.cuh"

// dst[i,t] = sum_k a[i,k,t] * b[0,k,t]
//   a:   [n, k, m]  (ne0=n, ne1=k, ne2=m)
//   b:   [1, k, m]  (broadcast over ne0)
//   dst: [n, m]
// One thread per output element (i, t); each loops over the (small) k axis.
static __global__ void weighted_sum_f32_cuda(
        const float * __restrict__ a, const float * __restrict__ b, float * __restrict__ dst,
        const int n, const int k, const int m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= n || t >= m) {
        return;
    }
    const float * __restrict__ a_t = a + (int64_t) t * n * k; // a[:, :, t]
    const float * __restrict__ b_t = b + (int64_t) t * k;      // b[0, :, t]
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        acc += a_t[i + (int64_t) kk * n] * b_t[kk];
    }
    dst[i + (int64_t) t * n] = acc;
}

void ggml_cuda_op_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // a
    const ggml_tensor * src1 = dst->src[1]; // b

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float         * dst_d = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n = (int) src0->ne[0];
    const int k = (int) src0->ne[1];
    const int m = (int) src0->ne[2];

    const dim3 block_nums((n + 255) / 256, m, 1);
    const dim3 block_dims(256, 1, 1);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    ggml_cuda_kernel_launch(weighted_sum_f32_cuda, launch_params, src0_d, src1_d, dst_d, n, k, m);
}
