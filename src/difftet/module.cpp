#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "render.h"
#include "tests.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("render_forward", &render_forward);
    m.def("render_backward", &render_backward);
    m.def("cartesian_to_bary", &cartesian_to_bary);
    // m.def("intersection", &intersection_torch);
    // m.def("intersection_backward", &intersection_backward_torch);
    // m.def("project", &project_torch);
    // m.def("project_backward", &project_backward_torch);
    // m.def("bary", &bary_torch);
    // m.def("bary_backward", &bary_backward_torch);
    // m.def("compute_color", &compute_color_torch);
    // m.def("compute_color_backward", &compute_color_backward_torch);
    // m.def("compute_opacity", &compute_opacity_torch);
    // m.def("compute_opacity_backward", &compute_opacity_backward_torch);
    // m.def("dist", &dist_torch);
    // m.def("dist_backward", &dist_backward_torch);
    // m.def("interpolate3_scalar", &interpolate3_scalar_torch);
    // m.def("interpolate3_scalar_backward", &interpolate3_scalar_backward_torch);
    // m.def("interpolate3_vector", &interpolate3_vector_torch);
    // m.def("interpolate3_vector_backward", &interpolate3_vector_backward_torch);
    // m.def("lerp_scalar", &lerp_scalar_torch);
    // m.def("lerp_scalar_backward", &lerp_scalar_backward_torch);
    // m.def("lerp_vector", &lerp_vector_torch);
    // m.def("lerp_vector_backward", &lerp_vector_backward_torch);
}