#include <torch/extension.h>
#include <map>
#include <string>
#include "util.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_backward(
    torch::Tensor grad_output,                                                                    // upstream gradient
    torch::Tensor vertices, torch::Tensor indices, torch::Tensor colors, torch::Tensor opacities, // triangle data
    torch::Tensor sorted_ids, torch::Tensor offsets,                                              // ids and offsets for sorted rendering
    torch::Tensor bary_transforms,                                                                // cartesian to barycentric transformation matrices
    torch::Tensor final_opacities, torch::Tensor ends,                                            // final opacities and indices
    int width, int height,                                                                        // image size
    const int tile_width, const int tile_height,                                                  // tile size
    const scalar early_stopping_threshold                                                         // remaining opacity at which to stop rendering
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::map<std::string, float>> render_forward(
    torch::Tensor vertices, torch::Tensor indices, torch::Tensor colors, torch::Tensor opacities, // input: vertices, indices, colors and opacities
    torch::Tensor depths,
    int width, int height,                       // image size
    const int tile_width, const int tile_height, // tile size
    const scalar early_stopping_threshold,       // remaining opacity at which to stop rendering
    bool disable_timing);

torch::Tensor cartesian_to_bary(torch::Tensor vertices, torch::Tensor indices);