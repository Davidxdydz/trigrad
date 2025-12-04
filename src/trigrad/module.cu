#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "render.h"
#include "tests.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("render_forward", &render_forward);
    m.def("render_backward", &render_backward);
    m.def("cartesian_to_bary", &cartesian_to_bary);
    m.def("interpolate3_scalar_torch", &interpolate3_scalar_torch);
    m.def("interpolate3_scalar_backward_torch", &interpolate3_scalar_backward_torch);
    m.def("interpolate3_vector_torch", &interpolate3_vector_torch);
    m.def("interpolate3_vector_backward_torch", &interpolate3_vector_backward_torch);
    m.def("barycentric_torch", &barycentric_torch);
    m.def("barycentric_backward_torch", &barycentric_backward_torch);
#ifdef DOUBLE_PRECISION
    m.attr("precision") = torch::kFloat64;
#else
    m.attr("precision") = torch::kFloat32;
#endif
}