#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor add_matrix_cu_forward(
    torch::Tensor A,
    torch::Tensor B);


torch::Tensor add_matrix_forward(
  torch::Tensor A,
  torch::Tensor B
){
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    return add_matrix_cu_forward(A, B);
}

std::vector<torch::Tensor> add_matrix_backward(
  torch::Tensor grad_out
){
    return {grad_out, grad_out};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &add_matrix_forward);
    m.def("backward", &add_matrix_backward);
}