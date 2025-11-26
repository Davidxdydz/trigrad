#include "render.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "util.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void touched_tiles(
    vec2 a, vec2 b, vec2 c,          // triangle vertices
    int width, int height,           // image size
    int tile_width, int tile_height, // tile size
    int tiles_x, int tiles_y,        // number of tiles in x and y directions
    int2 &mins, int2 &maxs           // output : min and max tile coordinates
)
{

    double v = cross2d(b - a, c - a);
    if (v == 0 || std::isnan(v) || std::isinf(v))
    {
        // triangle is too small, return empty tile range
        mins = {0, -1};
        maxs = {0, -1};
        return;
    }
    scalar minx = std::min({a.x, b.x, c.x});
    scalar miny = std::min({a.y, b.y, c.y});
    scalar maxx = std::max({a.x, b.x, c.x});
    scalar maxy = std::max({a.y, b.y, c.y});
    mins.x = std::clamp((minx * width) / tile_width, (scalar)0, (scalar)(tiles_x - 1));
    mins.y = std::clamp((miny * height) / tile_height, (scalar)0, (scalar)(tiles_y - 1));
    maxs.x = std::clamp((maxx * width) / tile_width, (scalar)0, (scalar)(tiles_x - 1));
    maxs.y = std::clamp((maxy * height) / tile_height, (scalar)0, (scalar)(tiles_y - 1));
}

__device__ vec3 apply_cart_to_bary(const scalar *bary_matrices, vec2 p)
{

    // p is in [0,1] range
    // m is a 3x3 matrix M, to be applied to p' = (p.x, p.y, 1)^T like M * p'
    // the result are barycentric coordinates
    vec3 result;
    result.x = bary_matrices[0] * p.x + bary_matrices[1] * p.y + bary_matrices[2];
    result.y = bary_matrices[3] * p.x + bary_matrices[4] * p.y + bary_matrices[5];
    result.z = bary_matrices[6] * p.x + bary_matrices[7] * p.y + bary_matrices[8];
    return result;
}

__global__ void count_per_tile_kernel(
    int *__restrict__ counts,                                           // output: how many triangles are in each tile
    const vec3 *__restrict__ vertices, const id3 *__restrict__ indices, // vertices and indices of the triangles
    int triangle_count,                                                 // number of triangles
    int width, int height,                                              // image size
    int tile_width, int tile_height                                     // tile size
)
{
    int2 mins, maxs;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= triangle_count)
        return;

    id3 tri = indices[id];

    int tiles_x = (width + tile_width - 1) / tile_width;
    int tiles_y = (height + tile_height - 1) / tile_height;
    touched_tiles(
        xy(vertices[tri.a]), xy(vertices[tri.b]), xy(vertices[tri.c]),
        width, height, tile_width, tile_height, tiles_x, tiles_y, mins, maxs);
    // TODO explicitly check the triangles against each potential tile, atm all in AABB are counted
    if (mins.y == -1)
        return;
    for (int i = mins.x; i <= maxs.x; i++)
    {
        for (int j = mins.y; j <= maxs.y; j++)
        {
            int index = j * tiles_x + i;
            atomicAdd(&counts[index], 1);
        }
    }
}

torch::Tensor count_per_tile(torch::Tensor vertices, torch::Tensor indices, int width, int height, const int tile_width, const int tile_height)
{
    int triangle_count = indices.size(0);
    const dim3 threads_per_block(256);
    const dim3 blocks((triangle_count + threads_per_block.x - 1) / threads_per_block.x);
    auto counts = torch::zeros({(height + tile_height - 1) / tile_height, (width + tile_width - 1) / tile_width}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    vertices = vertices.contiguous();
    indices = indices.contiguous();
    int *counts_ptr = counts.mutable_data_ptr<int>();
    count_per_tile_kernel<<<blocks, threads_per_block>>>(
        counts_ptr,
        const_vec3(vertices), const_id3(indices),
        triangle_count,
        width, height,
        tile_width, tile_height);
    return counts;
}

__global__ void fill_per_tile_lists_kernel(
    int *__restrict__ var_offsets,                                         // mutable data pointer to the offsets, this is used as an internal counter
    int *__restrict__ per_tile_list, scalar *__restrict__ per_tile_depths, // output: indices to triangles and depths per tile
    const vec3 *__restrict__ vertices, const id3 *__restrict__ indices,    // vertices and indices of the triangles
    const scalar *__restrict__ depths,                                     // depth of the triangles
    int triangle_count,                                                    // number of triangles
    int width, int height,                                                 // image size
    const int tile_width, const int tile_height                            // tile size
)
{
    int2 mins, maxs;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= triangle_count)
        return;

    id3 tri = indices[id];
    scalar depth = depths[id];
    int tiles_x = (width + tile_width - 1) / tile_width;
    int tiles_y = (height + tile_height - 1) / tile_height;
    touched_tiles(
        xy(vertices[tri.a]), xy(vertices[tri.b]), xy(vertices[tri.c]),
        width, height, tile_width, tile_height, tiles_x, tiles_y, mins, maxs);
    if (mins.y == -1)
        return;
    for (int i = mins.x; i <= maxs.x; i++)
    {
        for (int j = mins.y; j <= maxs.y; j++)
        {
            int index = j * tiles_x + i;
            int my_index = atomicAdd(&var_offsets[index], 1);
            per_tile_list[my_index] = id;
            per_tile_depths[my_index] = depth;
        }
    }
}

void sort_depth_ranges(
    scalar *keys_out, int *values_out,                               // output: sorted keys and values
    const scalar *keys_in, const int *values_in, const int *offsets, // input: keys and values to sort, offsets for the segments like [0,4,8] for 2 segments
    int num_items, int num_segments                                  // number total items, number of segments
)
{
    void *tmp_storage = nullptr;
    size_t tmp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(tmp_storage, tmp_storage_bytes, keys_in, keys_out, values_in, values_out, num_items, num_segments, offsets, offsets + 1);
    cudaMalloc(&tmp_storage, tmp_storage_bytes);
    cub::DeviceSegmentedSort::SortPairs(tmp_storage, tmp_storage_bytes, keys_in, keys_out, values_in, values_out, num_items, num_segments, offsets, offsets + 1);
    cudaFree(tmp_storage);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> per_tile_lists(torch::Tensor vertices, torch::Tensor indices, torch::Tensor depths, int width, int height, const int tile_width, const int tile_height)
{
    auto offsets = count_per_tile(vertices, indices, width, height, tile_width, tile_height).view(-1);
    int tile_count = offsets.size(0);
    int total_size = offsets.sum().item<int>();
    int triangle_count = indices.size(0);
    torch::cumsum_out(offsets, offsets, 0);
    auto var_offsets = torch::zeros({tile_count + 1}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    var_offsets.slice(0, 1, tile_count + 1) = offsets;
    offsets = var_offsets.clone();
    dim3 threads_per_block(256);
    dim3 blocks((triangle_count + threads_per_block.x - 1) / threads_per_block.x);
    auto per_tile_list = torch::empty({total_size}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    auto per_tile_depths = torch::empty({total_size}, torch::TensorOptions(torchscalar).device(torch::kCUDA));
    fill_per_tile_lists_kernel<<<blocks, threads_per_block>>>(
        mutable_int(var_offsets),
        mutable_int(per_tile_list), mutable_scalar(per_tile_depths),
        const_vec3(vertices),
        const_id3(indices), const_scalar(depths), triangle_count, width, height, tile_width, tile_height);

    return {per_tile_list, per_tile_depths, offsets};
}

__global__ void render_forward_kernel(
    color3 *__restrict__ image, scalar *final_opacity, int *__restrict__ ends,                                // output: color, final opacity, index of last rendered triangle per pixel
    const scalar *bary_transforms,                                                                            // cartesian to barycentric matrices, 3x3 per triangle
    const id3 *__restrict__ indices, const color3 *__restrict__ colors, const scalar *__restrict__ opacities, // triangle indices, colors and opacities
    const int *__restrict__ per_tile_list, const int *__restrict__ offsets,                                   // values from sorting
    int width, int height,
    const scalar early_stopping_threshold

)
{
    int tile_index = blockIdx.y * gridDim.x + blockIdx.x;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    bool stopped = false;
    bool oob = false;
    if (iy >= height || ix >= width)
    {
        stopped = true;
        oob = true;
    }
    int index = iy * width + ix;
    int start = offsets[tile_index];
    int end = offsets[tile_index + 1];
    vec2 pos = {((scalar)ix + 0.5f) / (scalar)width, ((scalar)iy + 0.5f) / (scalar)height};
    constexpr const int n_prefetch = 64;
    int local_index = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ scalar bary_transforms_shared[3 * 3 * n_prefetch];
    __shared__ color3 colors_shared[n_prefetch * 3];
    __shared__ scalar opacities_shared[n_prefetch * 3];
    // TODO maybe store colors together with transforms
    // TODO maybe store per vertex data contiguously
    scalar alpha = 1.0f;
    color3 total_color = {0, 0, 0};
    int end_out = start;
    for (int j = start; j < end; j += n_prefetch)
    {
        int num_done = __syncthreads_count(stopped);
        if (num_done == blockDim.x * blockDim.y)
            break;
        int prefetch_index = j + local_index;
        if (prefetch_index < end && local_index < n_prefetch)
        {
            int tri_index = per_tile_list[prefetch_index];
            id3 tri = indices[tri_index];
            for (int k = 0; k < 9; k++)
            {
                bary_transforms_shared[local_index * 9 + k] = bary_transforms[tri_index * 9 + k];
            }
            colors_shared[local_index * 3 + 0] = colors[tri.a];
            colors_shared[local_index * 3 + 1] = colors[tri.b];
            colors_shared[local_index * 3 + 2] = colors[tri.c];
            opacities_shared[local_index * 3 + 0] = opacities[tri.a];
            opacities_shared[local_index * 3 + 1] = opacities[tri.b];
            opacities_shared[local_index * 3 + 2] = opacities[tri.c];
        }
        __syncthreads();
        for (int k = 0; k < n_prefetch && j + k < end && !stopped; k++)
        {
            vec3 bary = apply_cart_to_bary(&bary_transforms_shared[k * 9], pos);
            if (bary.x < 0.0 || bary.y < 0.0 || bary.z < 0.0)
                continue;
            if (bary.x == 0 && bary.y == 0 && bary.z == 0)
                continue;
            scalar opacity = interpolate3(bary, opacities_shared[k * 3], opacities_shared[k * 3 + 1], opacities_shared[k * 3 + 2]);
            opacity = std::clamp(opacity, 0.0, max_opacity);
            color3 color = interpolate3(bary, colors_shared[k * 3], colors_shared[k * 3 + 1], colors_shared[k * 3 + 2]);
            total_color = total_color + alpha * opacity * color;
            alpha *= (1 - opacity);
            end_out = j + k + 1;
            if (alpha < early_stopping_threshold)
                stopped = true;
        }
    }
    if (!oob)
    {
        ends[index] = end_out;
        final_opacity[index] = alpha;
        image[index] = total_color;
    }
}

__global__ void cartesian_to_bary_kernel(
    scalar *__restrict__ output,                                        // output: barycentric transformation matrices nx3x3
    const vec3 *__restrict__ vertices, const id3 *__restrict__ indices, // vertices and indices of the triangles
    int num_triangles                                                   // number of triangles
)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_triangles)
        return;
    id3 i = indices[id];
    vec2 a = xy(vertices[i.a]);
    vec2 b = xy(vertices[i.b]);
    vec2 c = xy(vertices[i.c]);
    int mid = id * 3 * 3;
    scalar denom = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
    if (denom == 0)
        denom = 1;

    output[mid + 0] = (b.y - c.y) / denom;
    output[mid + 1] = (c.x - b.x) / denom;
    output[mid + 2] = (b.x * c.y - c.x * b.y) / denom;
    output[mid + 3] = (c.y - a.y) / denom;
    output[mid + 4] = (a.x - c.x) / denom;
    output[mid + 5] = (c.x * a.y - a.x * c.y) / denom;
    output[mid + 6] = (a.y - b.y) / denom;
    output[mid + 7] = (b.x - a.x) / denom;
    output[mid + 8] = (a.x * b.y - b.x * a.y) / denom;
}

torch::Tensor cartesian_to_bary(torch::Tensor vertices, torch::Tensor indices)
{
    int triangle_count = indices.size(0);
    auto output = torch::zeros({triangle_count, 3, 3}, torch::TensorOptions(torchscalar).device(torch::kCUDA));
    vertices = vertices.contiguous();
    indices = indices.contiguous();
    const dim3 threads_per_block(256);
    const dim3 blocks((triangle_count + threads_per_block.x - 1) / threads_per_block.x);
    cartesian_to_bary_kernel<<<blocks, threads_per_block>>>(
        mutable_scalar(output),
        const_vec3(vertices),
        const_id3(indices),
        triangle_count);
    return output;
}
struct CudaTimer
{
    std::map<std::string, float> &timings;
    cudaEvent_t _start, _stop;
    bool active = false;
    std::string current_name;
    bool disable;

    CudaTimer(std::map<std::string, float> &timings_, bool disable_ = false) : timings(timings_), disable(disable_)
    {
        if (disable)
            return;
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }
    void stop()
    {
        if (disable)
            return;
        if (!active)
            return;
        cudaEventRecord(_stop);
        cudaEventSynchronize(_stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, _start, _stop);
        timings[current_name] = milliseconds / 1000.0f;
        active = false;
    }
    void start(std::string name)
    {
        if (disable)
            return;
        if (active)
            stop();
        cudaEventRecord(_start);
        current_name = name;
        active = true;
    }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::map<std::string, float>> render_forward(
    torch::Tensor vertices, torch::Tensor indices, torch::Tensor colors, torch::Tensor opacities, // input: vertices, indices, colors and opacities
    torch::Tensor depths,
    int width, int height,                       // image size
    const int tile_width, const int tile_height, // tile size
    const scalar early_stopping_threshold,       // remaining opacity at which to stop rendering
    bool disable_timing)
{
    std::map<std::string, float> timings;
    CudaTimer timer(timings, disable_timing);

    timer.start("init");
    vertices = vertices.contiguous();
    indices = indices.contiguous();
    colors = colors.contiguous();
    opacities = opacities.contiguous();
    depths = depths.contiguous();
    auto image = torch::zeros({height, width, 3}, torch::TensorOptions(torchscalar).device(torch::kCUDA));
    auto ends = torch::zeros({height, width}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));
    auto final_opacity = torch::zeros({height, width}, torch::TensorOptions(torchscalar).device(torch::kCUDA));

    timer.start("per_tile_lists");
    auto [ids, per_tile_depths, offsets] = per_tile_lists(vertices, indices, depths, width, height, tile_width, tile_height);

    timer.start("sorting");
    auto ids_out = torch::empty_like(ids);
    auto depths_out = torch::empty_like(per_tile_depths);
    sort_depth_ranges(
        mutable_scalar(depths_out), mutable_int(ids_out),
        const_scalar(per_tile_depths), const_int(ids), const_int(offsets),
        ids.size(0), offsets.size(0) - 1);
    ids = ids_out;

    timer.start("bary_transforms");
    auto bary_transforms = cartesian_to_bary(vertices, indices);
    timer.start("rasterization");
    const dim3 threads_per_block(tile_width, tile_height);
    const dim3 blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);
    render_forward_kernel<<<blocks, threads_per_block>>>(
        mutable_color3(image), mutable_scalar(final_opacity), mutable_int(ends),
        const_scalar(bary_transforms),
        const_id3(indices), const_color3(colors), const_scalar(opacities),
        const_int(ids), const_int(offsets),
        width, height,
        early_stopping_threshold);
    timer.stop();
    return {image, ids, offsets, bary_transforms, final_opacity, ends, timings};
}

__global__ void render_backward_kernel(
    vec3 *__restrict__ grad_vertices, color3 *__restrict__ grad_colors, scalar *__restrict__ grad_opacities,                                     // output: gradients
    const color3 *__restrict__ grad_output,                                                                                                      // upstream gradient
    const scalar *__restrict__ final_opacities, const int *__restrict__ ends,                                                                    // final opacities and indices
    const int *__restrict__ sorted_ids, const int *__restrict__ offsets,                                                                         // ids and offsets for sorted rendering
    const scalar *__restrict__ bary_transforms,                                                                                                  // cartesian to barycentric transformation matrices
    const vec3 *__restrict__ vertices, const id3 *__restrict__ indices, const color3 *__restrict__ colors, const scalar *__restrict__ opacities, // triangle data
    int width, int height,                                                                                                                       // image size
    const scalar early_stopping_threshold                                                                                                        // remaining opacity at which to stop rendering
)
{
    int tile_index = blockIdx.y * gridDim.x + blockIdx.x;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= height || ix >= width)
        return;
    int index = iy * width + ix;
    int start = offsets[tile_index];
    int end = ends[index];
    scalar x = ((scalar)ix + 0.5f) / (scalar)width;
    scalar y = ((scalar)iy + 0.5f) / (scalar)height;
    scalar alpha = final_opacities[index];
    vec3 s = {0, 0, 0};

    for (int k = end - 1; k >= start; k--)
    {
        int id = sorted_ids[k];
        id3 i3 = indices[id];
        vec3 bary = apply_cart_to_bary(&bary_transforms[id * 9], {x, y});
        if (bary.x < 0.0 || bary.y < 0.0 || bary.z < 0.0)
            continue;
        if (bary.x == 0 && bary.y == 0 && bary.z == 0)
            continue;

        scalar opacity = interpolate3(bary, opacities[i3.a], opacities[i3.b], opacities[i3.c]);
        opacity = std::clamp(opacity, 0.0, max_opacity);

        color3 color = interpolate3(bary, colors[i3.a], colors[i3.b], colors[i3.c]);
        // opacity gradient
        scalar d_do = component_sum((color * alpha - s) / (1 - opacity) * grad_output[index]);

        auto [d_do_do_dbary, d_do1, d_do2, d_do3, d_do_do_dw] = interpolate3_backward(d_do, bary, opacities[i3.a], opacities[i3.b], opacities[i3.c], {1, 1, 1});
        atomicAdd(&grad_opacities[i3.a], d_do1);
        atomicAdd(&grad_opacities[i3.b], d_do2);
        atomicAdd(&grad_opacities[i3.c], d_do3);

        alpha /= (1.0 - opacity);

        s += color * opacity * alpha;

        // color gradient
        color3 d_rgb_dc = alpha * opacity * grad_output[index];
        auto [drgb_dc_dc_dbary, drgb_dc1, drgb_dc2, drgb_dc3, drgb_dc_dc_dw] = interpolate3_backward(d_rgb_dc, bary, colors[i3.a], colors[i3.b], colors[i3.c], {1, 1, 1});

        atomicAdd3(&grad_colors[i3.a], drgb_dc1);
        atomicAdd3(&grad_colors[i3.b], drgb_dc2);
        atomicAdd3(&grad_colors[i3.c], drgb_dc3);

        // vertex gradient
        vec3 d_dbary = d_do_do_dbary + drgb_dc_dc_dbary;
        auto [d_dvaxy, d_dvbxy, d_dvcxy, d_dp] = barycentric_backward(
            d_dbary,
            xy(vertices[i3.a]), xy(vertices[i3.b]), xy(vertices[i3.c]),
            {x, y});
        vec3 d_dva = {d_dvaxy.x, d_dvaxy.y, 0};
        vec3 d_dvb = {d_dvbxy.x, d_dvbxy.y, 0};
        vec3 d_dvc = {d_dvcxy.x, d_dvcxy.y, 0};

        atomicAdd3(&grad_vertices[i3.a], d_dva);
        atomicAdd3(&grad_vertices[i3.b], d_dvb);
        atomicAdd3(&grad_vertices[i3.c], d_dvc);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_backward(
    torch::Tensor grad_output,                                                                    // upstream gradient
    torch::Tensor vertices, torch::Tensor indices, torch::Tensor colors, torch::Tensor opacities, // triangle data
    torch::Tensor sorted_ids, torch::Tensor offsets,                                              // ids and offsets for sorted rendering
    torch::Tensor bary_transforms,                                                                // cartesian to barycentric transformation matrices
    torch::Tensor final_opacities, torch::Tensor ends,                                            // final opacities and indices
    int width, int height,                                                                        // image size
    const int tile_width, const int tile_height,                                                  // tile size
    const scalar early_stopping_threshold                                                         // remaining opacity at which to stop rendering
)
{
    colors = colors.contiguous();
    vertices = vertices.contiguous();
    indices = indices.contiguous();
    opacities = opacities.contiguous();
    auto grad_vertices = torch::zeros_like(vertices);
    auto grad_colors = torch::zeros_like(colors);
    auto grad_opacities = torch::zeros_like(opacities);
    const dim3 threads_per_block(tile_width, tile_height);
    const dim3 blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);
    render_backward_kernel<<<blocks, threads_per_block>>>(
        mutable_vec3(grad_vertices), mutable_color3(grad_colors), mutable_scalar(grad_opacities),
        const_color3(grad_output),
        const_scalar(final_opacities), const_int(ends),
        const_int(sorted_ids), const_int(offsets),
        const_scalar(bary_transforms),
        const_vec3(vertices), const_id3(indices), const_color3(colors), const_scalar(opacities),
        width, height,
        early_stopping_threshold);
    return {grad_vertices, grad_colors, grad_opacities};
}
