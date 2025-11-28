#include <cuda_runtime.h>
#include <torch/extension.h>
#include "util.h"

torch::Tensor barycentric_torch(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor p);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> barycentric_backward_torch(torch::Tensor d_db, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor p);

torch::Tensor interpolate3_scalar_torch(torch::Tensor bary, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor w);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> interpolate3_scalar_backward_torch(torch::Tensor d_dinter, torch::Tensor bary, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor w);

torch::Tensor interpolate3_vector_torch(torch::Tensor bary, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor w);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> interpolate3_vector_backward_torch(torch::Tensor d_dinter, torch::Tensor bary, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor w);
