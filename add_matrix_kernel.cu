#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ __forceinline__ scalar_t identity(scalar_t z) {
    return z;
}


template <typename scalar_t>
__global__ void add_matrix_kernel_forward(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> A,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> B,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> out
){    
    const int n = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < A.size(1)){
        out[n][c] = identity(A[n][c] + B[n][c]);
    }
}


torch::Tensor add_matrix_cu_forward(
  torch::Tensor A,
  torch::Tensor B
){
    torch::Tensor out = torch::zeros_like(A);

    const int n_row = A.size(0);
    const int n_col = A.size(1);
    const int threads = 1024;
    const dim3 blocks((n_col + threads - 1) / threads, n_row); // to cover all elements
    
    // instantiate kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "add_matrix_cu_forward", 
    ([&] {
        add_matrix_kernel_forward<scalar_t><<<blocks, threads>>>(
            A.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            B.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            out.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    })
    );
    return out;
}