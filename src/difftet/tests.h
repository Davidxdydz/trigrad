#include <cuda_runtime.h>
#include <torch/extension.h>
#include "util.h"

// std::tuple<torch::Tensor, torch::Tensor> intersection_torch(torch::Tensor p0, torch::Tensor p1, torch::Tensor q0, torch::Tensor q1);
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> intersection_backward_torch(scalar d_t0, scalar d_t1, torch::Tensor p0, torch::Tensor p1, torch::Tensor q0, torch::Tensor q1);

// torch::Tensor project_torch(torch::Tensor v, torch::Tensor projection_matrix);
// torch::Tensor project_backward_torch(torch::Tensor d_dpersp, torch::Tensor v, torch::Tensor projection_matrix);

torch::Tensor barycentric_torch(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor p);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> barycentric_backward_torch(torch::Tensor d_db, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor p);

// torch::Tensor compute_color_torch(torch::Tensor o0, torch::Tensor o1, torch::Tensor c0, torch::Tensor c1, torch::Tensor d);
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> compute_color_backward_torch(torch::Tensor d_dcolor, torch::Tensor o0, torch::Tensor o1, torch::Tensor c0, torch::Tensor c1, torch::Tensor d);

// torch::Tensor compute_opacity_torch(torch::Tensor o0, torch::Tensor o1, torch::Tensor d, bool merge_depth_opacity);
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> compute_opacity_backward_torch(torch::Tensor d_dopacity, torch::Tensor o0, torch::Tensor o1, torch::Tensor d, bool merge_depth_opacity);

// torch::Tensor dist_torch(torch::Tensor p0, torch::Tensor p1);
// std::tuple<torch::Tensor, torch::Tensor> dist_backward_torch(torch::Tensor d_dd, torch::Tensor p0, torch::Tensor p1);

torch::Tensor interpolate3_scalar_torch(torch::Tensor bary, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor w);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> interpolate3_scalar_backward_torch(torch::Tensor d_dinter, torch::Tensor bary, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor w);

torch::Tensor interpolate3_vector_torch(torch::Tensor bary, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor w);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> interpolate3_vector_backward_torch(torch::Tensor d_dinter, torch::Tensor bary, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor w);

// torch::Tensor lerp_scalar_torch(torch::Tensor a, torch::Tensor b, torch::Tensor t, torch::Tensor w);
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> lerp_scalar_backward_torch(torch::Tensor d_dlerp, torch::Tensor a, torch::Tensor b, torch::Tensor t, torch::Tensor w);

// torch::Tensor lerp_vector_torch(torch::Tensor a, torch::Tensor b, torch::Tensor t, torch::Tensor w);
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> lerp_vector_backward_torch(torch::Tensor d_dlerp, torch::Tensor a, torch::Tensor b, torch::Tensor t, torch::Tensor w);