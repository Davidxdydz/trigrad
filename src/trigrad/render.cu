#include "render.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "util.h"
#include <cooperative_groups.h>
#include <limits>

namespace cg = cooperative_groups;

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
        timings[current_name] = milliseconds / 1000.0;
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

__device__ void touched_tiles(
    vec2 a, vec2 b, vec2 c,          // triangle vertices
    int width, int height,           // image size
    int tile_width, int tile_height, // tile size
    int tiles_x, int tiles_y,        // number of tiles in x and y directions
    int2 &mins, int2 &maxs           // output : min and max tile coordinates
)
{

    scalar v = cross2d(b - a, c - a);
    if (std::abs(v) < eff_zero || std::isnan(v) || std::isinf(v))
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
    mins.x = std::clamp((scalar)(((minx + 1) * 0.5 * width) / tile_width), (scalar)0, (scalar)(tiles_x - 1));
    mins.y = std::clamp((scalar)(((miny + 1) * 0.5 * height) / tile_height), (scalar)0, (scalar)(tiles_y - 1));
    maxs.x = std::clamp((scalar)(((maxx + 1) * 0.5 * width) / tile_width), (scalar)0, (scalar)(tiles_x - 1));
    maxs.y = std::clamp((scalar)(((maxy + 1) * 0.5 * height) / tile_height), (scalar)0, (scalar)(tiles_y - 1));
}

__device__ vec3 apply_cart_to_bary(const scalar *bary_matrices, vec2 p)
{

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
    const vec4 *__restrict__ vertices, const id3 *__restrict__ indices, // vertices and indices of the triangles
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
        const_vec4(vertices), const_id3(indices),
        triangle_count,
        width, height,
        tile_width, tile_height);
    return counts;
}
__device__ inline void compute_plane(const vec3 &a, const vec3 &b, const vec3 &c,
                                     scalar &A, scalar &B, scalar &C)
{
    scalar det = a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y);
    scalar id = 1.0 / det;

    A = (a.z * (b.y - c.y) + b.z * (c.y - a.y) + c.z * (a.y - b.y)) * id;
    B = (a.x * (b.z - c.z) + b.x * (c.z - a.z) + c.x * (a.z - b.z)) * id;
    C = (a.x * (b.y * c.z - c.y * b.z) +
         b.x * (c.y * a.z - a.y * c.z) +
         c.x * (a.y * b.z - b.y * a.z)) *
        id;
}

__device__ inline scalar plane_depth(scalar A, scalar B, scalar C, const vec2 &p)
{
    return A * p.x + B * p.y + C;
}

__device__ inline int clip_polygon_to_halfspace(
    const vec2 *in_p, int n_in, vec2 *out_p,
    scalar nx, scalar ny, scalar d)
{
    int n_out = 0;

    vec2 p0 = in_p[n_in - 1];
    scalar d0 = nx * p0.x + ny * p0.y - d;
    bool in0 = d0 >= 0;

    for (int i = 0; i < n_in; i++)
    {
        vec2 p1 = in_p[i];
        scalar d1 = nx * p1.x + ny * p1.y - d;
        bool in1 = d1 >= 0;

        if (in1 != in0)
        {
            scalar t = d0 / (d0 - d1);
            out_p[n_out++] = {p0.x + t * (p1.x - p0.x),
                              p0.y + t * (p1.y - p0.y)};
        }
        if (in1)
            out_p[n_out++] = p1;

        p0 = p1;
        d0 = d1;
        in0 = in1;
    }
    return n_out;
}

__device__ inline bool clip_rect(vec2 *poly, int &n, vec2 *buf,
                                 scalar nx, scalar ny, scalar d)
{
    n = clip_polygon_to_halfspace(poly, n, buf, nx, ny, d);
    if (n <= 0)
        return false;
    for (int i = 0; i < n; i++)
        poly[i] = buf[i];
    return true;
}

__global__ void fill_per_tile_lists_kernel(
    int *__restrict__ var_offsets,
    int *__restrict__ per_tile_list,
    scalar *__restrict__ per_tile_depths,
    const vec4 *__restrict__ vertices,
    const id3 *__restrict__ indices,
    int tri_count,
    int width, int height,
    const int tw, const int th)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tri_count)
        return;

    id3 tri = indices[tid];
    int tiles_x = (width + tw - 1) / tw;
    int tiles_y = (height + th - 1) / th;

    vec3 a = {vertices[tri.a].x, vertices[tri.a].y, vertices[tri.a].z};
    vec3 b = {vertices[tri.b].x, vertices[tri.b].y, vertices[tri.b].z};
    vec3 c = {vertices[tri.c].x, vertices[tri.c].y, vertices[tri.c].z};

    int2 mins, maxs;
    touched_tiles(xy(a), xy(b), xy(c),
                  width, height, tw, th,
                  tiles_x, tiles_y,
                  mins, maxs);
    if (mins.y == -1)
        return;

    scalar A, B, Cc;
    compute_plane(a, b, c, A, B, Cc);

    for (int tx = mins.x; tx <= maxs.x; tx++)
        for (int ty = mins.y; ty <= maxs.y; ty++)
        {
            int tile_i = ty * tiles_x + tx;
            int out_i = atomicAdd(&var_offsets[tile_i], 1);

            per_tile_list[out_i] = tid;

            scalar xmin = (scalar)(tx * tw) / width * 2 - 1;
            scalar xmax = (scalar)((tx + 1) * tw) / width * 2 - 1;
            scalar ymin = (scalar)(ty * th) / height * 2 - 1;
            scalar ymax = (scalar)((ty + 1) * th) / height * 2 - 1;

            vec2 poly[7], buf[7];
            int n = 3;
            poly[0] = xy(a);
            poly[1] = xy(b);
            poly[2] = xy(c);

            if (!clip_rect(poly, n, buf, +1, 0, xmin) ||
                !clip_rect(poly, n, buf, -1, 0, -xmax) ||
                !clip_rect(poly, n, buf, 0, +1, ymin) ||
                !clip_rect(poly, n, buf, 0, -1, -ymax))
            {
                per_tile_depths[out_i] = 1e30;
                continue;
            }

            scalar md = std::numeric_limits<scalar>::max();
#pragma unroll
            for (int i = 0; i < n; i++)
                md = std::min(md, plane_depth(A, B, Cc, poly[i]));

            per_tile_depths[out_i] = md;
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
    cub::DeviceSegmentedSort::StableSortPairs(tmp_storage, tmp_storage_bytes, keys_in, keys_out, values_in, values_out, num_items, num_segments, offsets, offsets + 1);
    // use torch to allocate temporary storage to use torch memory pool
    tmp_storage = torch::empty({int(tmp_storage_bytes)}, torch::TensorOptions(torch::kUInt8).device(torch::kCUDA)).data_ptr();
    cub::DeviceSegmentedSort::StableSortPairs(tmp_storage, tmp_storage_bytes, keys_in, keys_out, values_in, values_out, num_items, num_segments, offsets, offsets + 1);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> per_tile_lists(torch::Tensor vertices, torch::Tensor indices, int width, int height, const int tile_width, const int tile_height)
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
        const_vec4(vertices),
        const_id3(indices),
        triangle_count, width, height, tile_width, tile_height);

    return {per_tile_list, per_tile_depths, offsets};
}

__global__ void cartesian_to_bary_kernel(
    scalar *__restrict__ output,                                        // output: barycentric transformation matrices nx3x3
    const vec4 *__restrict__ vertices, const id3 *__restrict__ indices, // vertices and indices of the triangles
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
    if (std::abs(denom) < eff_zero)
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
__device__ inline void insert_sorted_topk(const int layers,
                                          int *topk,
                                          scalar *topk_depths,
                                          int &topk_idx,
                                          scalar pixel_depth,
                                          int value,
                                          bool ignore_sort = false)
{
    if (ignore_sort)
    {
        if (topk_idx < layers)
        {
            topk[topk_idx] = value;
            topk_depths[topk_idx] = pixel_depth;
            topk_idx++;
        }
        return;
    }
    if (topk_idx == layers)
    {
        if (pixel_depth >= topk_depths[layers - 1])
            return;
    }
    int idx = (topk_idx < layers) ? topk_idx : (layers - 1);
    while (idx > 0 && pixel_depth < topk_depths[idx - 1])
    {
        topk_depths[idx] = topk_depths[idx - 1];
        topk[idx] = topk[idx - 1];
        idx--;
    }
    topk_depths[idx] = pixel_depth;
    topk[idx] = value;

    if (topk_idx < layers)
        topk_idx++;
}

__device__ inline int select_next_k_in_tile(
    const int K,
    int *out_j,
    scalar *out_depths,
    const scalar lastDepth,
    const int lastJ,
    const int start,
    const int end,
    const int *__restrict__ per_tile_list,
    const scalar *__restrict__ bary_transforms,
    const vec4 *__restrict__ vertices,
    const id3 *__restrict__ indices,
    const vec2 pos)
{
    int topk_idx = 0;
    for (int i = 0; i < K; ++i)
        out_depths[i] = std::numeric_limits<scalar>::infinity();

    for (int j = start; j < end; ++j)
    {
        int tri_lookup = per_tile_list[j];
        id3 tri = indices[tri_lookup];

        vec3 ws = {vertices[tri.a].w, vertices[tri.b].w, vertices[tri.c].w};
        vec3 bary = apply_cart_to_bary(&bary_transforms[tri_lookup * 9], pos);

        if (bary.x < eff_zero || bary.y < eff_zero || bary.z < eff_zero)
            continue;

        scalar pixel_depth = interpolate3(bary, vertices[tri.a].z, vertices[tri.b].z, vertices[tri.c].z, ws);

        if (pixel_depth < lastDepth)
            continue;
        if (pixel_depth == lastDepth && j <= lastJ)
            continue;

        insert_sorted_topk(K, out_j, out_depths, topk_idx, pixel_depth, j, false);
    }

    return topk_idx;
}

__device__ inline void process_forward(
    int j,
    const int *__restrict__ per_tile_list,
    const scalar *__restrict__ bary_transforms,
    const vec4 *__restrict__ vertices,
    const id3 *__restrict__ indices,
    const color3 *__restrict__ colors,
    const scalar *__restrict__ opacities,
    const vec2 pos,
    scalar &alpha,
    color3 &total_color,
    scalar &total_depth,
    scalar &total_weight,
    int &end_out,
    int &blended_count)
{
    int tri_lookup = per_tile_list[j];
    id3 tri = indices[tri_lookup];
    vec3 ws = {vertices[tri.a].w, vertices[tri.b].w, vertices[tri.c].w};
    vec3 bary = apply_cart_to_bary(&bary_transforms[tri_lookup * 9], pos);

    if (bary.x < eff_zero || bary.y < eff_zero || bary.z < eff_zero)
        return;

    scalar opacity = interpolate3(bary, opacities[tri.a], opacities[tri.b], opacities[tri.c], ws);
    opacity = std::clamp(opacity, scalar(0.0), max_opacity);
    color3 color = interpolate3(bary, colors[tri.a], colors[tri.b], colors[tri.c], ws);
    vec3 inv_w = scalar(1.0) / ws;
    scalar depth = interpolate3(bary, inv_w.x, inv_w.y, inv_w.z);
    total_depth = total_depth + alpha * opacity * depth;
    total_weight = total_weight + alpha * opacity;
    total_color = total_color + alpha * opacity * color;
    alpha *= (1 - opacity);

    end_out++;
    blended_count++;
}

__device__ inline void process_backward(
    int j,
    const int *__restrict__ sorted_ids,
    const scalar *__restrict__ bary_transforms,
    const vec4 *__restrict__ vertices,
    const id3 *__restrict__ indices,
    const color3 *__restrict__ colors,
    const scalar *__restrict__ opacities,
    const vec2 pos,
    const color4 *__restrict__ grad_output,
    const scalar *__restrict__ grad_depthmap,
    int index,
    scalar final_alpha,
    scalar &alpha,
    vec3 &s,
    vec4 *__restrict__ grad_vertices,
    color3 *__restrict__ grad_colors,
    scalar *__restrict__ grad_opacities,
    scalar &oa,  // sum_{i=k+1}^n o_i * alpha_i
    scalar &zoa, // sum_{i=k+1}^n z_i * o_i * alpha_i
    scalar final_weight,
    scalar final_depth)
{
    int tri_index = sorted_ids[j];
    id3 tri = indices[tri_index];
    vec3 ws = {vertices[tri.a].w, vertices[tri.b].w, vertices[tri.c].w};
    vec3 bary = apply_cart_to_bary(&bary_transforms[tri_index * 9], pos);
    if (bary.x < eff_zero || bary.y < eff_zero || bary.z < eff_zero)
        return;

    scalar opacity = interpolate3(bary, opacities[tri.a], opacities[tri.b], opacities[tri.c], ws);
    opacity = std::clamp(opacity, scalar(0.0), max_opacity);
    color3 color = interpolate3(bary, colors[tri.a], colors[tri.b], colors[tri.c], ws);
    vec3 inv_w = scalar(1.0) / ws;
    scalar z = interpolate3(bary, inv_w.x, inv_w.y, inv_w.z);

    scalar drgb_do = component_sum((color * alpha - s) / (1 - opacity) * grad_output[index].rgb()) + grad_output[index].a * (-final_alpha / (1 - opacity));
    alpha /= (scalar(1.0) - opacity);
    s += color * opacity * alpha;
    scalar ddw_do = alpha - oa / (scalar(1.0) - opacity);      // d total depth weight / d opacity
    scalar dud_do = alpha * z - zoa / (scalar(1.0) - opacity); // d unnormalized depth / d opacity
    oa = oa + opacity * alpha;
    zoa = zoa + z * opacity * alpha;

    scalar dd_do = (dud_do - final_depth * ddw_do) / final_weight * grad_depthmap[index];
    if (final_weight < eff_zero)
        dd_do = scalar(0.0);
    scalar d_do = drgb_do + dd_do;
    auto [d_do_do_dbary, drgb_do1, drgb_do2, drgb_do3, drgb_do_do_dw] =
        interpolate3_backward(d_do, bary, opacities[tri.a], opacities[tri.b], opacities[tri.c], ws);

    atomicAdd(&grad_opacities[tri.a], drgb_do1);
    atomicAdd(&grad_opacities[tri.b], drgb_do2);
    atomicAdd(&grad_opacities[tri.c], drgb_do3);

    color3 drgb_dc = alpha * opacity * grad_output[index].rgb();
    auto [drgb_dc_dc_dbary, drgb_dc1, drgb_dc2, drgb_dc3, drgb_dc_dc_dw] =
        interpolate3_backward(drgb_dc, bary, colors[tri.a], colors[tri.b], colors[tri.c], ws);

    atomicAdd3(&grad_colors[tri.a], drgb_dc1);
    atomicAdd3(&grad_colors[tri.b], drgb_dc2);
    atomicAdd3(&grad_colors[tri.c], drgb_dc3);
    scalar dd_dz = alpha * opacity * grad_depthmap[index] / final_weight;
    if (final_weight < eff_zero)
        dd_dz = scalar(0.0);
    auto [dd_dz_dz_dbary, dd_dinv_wx, dd_dinv_wy, dd_dinv_wz, dd_dz_dz_dw] =
        interpolate3_backward(dd_dz, bary, inv_w.x, inv_w.y, inv_w.z);

    scalar dd_dwx = -dd_dinv_wx / (ws.x * ws.x);
    scalar dd_dwy = -dd_dinv_wy / (ws.y * ws.y);
    scalar dd_dwz = -dd_dinv_wz / (ws.z * ws.z);

    vec3 d_dbary = d_do_do_dbary + drgb_dc_dc_dbary + dd_dz_dz_dbary;
    auto [drgb_dva_xy, drgb_dvb_xy, drgb_dvc_xy, drgb_dp] =
        barycentric_backward(d_dbary, xy(vertices[tri.a]), xy(vertices[tri.b]), xy(vertices[tri.c]), pos);
    vec3 drgb_dw = drgb_do_do_dw + drgb_dc_dc_dw + dd_dz_dz_dw;
    vec4 drgb_dva = {drgb_dva_xy.x, drgb_dva_xy.y, scalar(0.0), drgb_dw.x + dd_dwx};
    vec4 drgb_dvb = {drgb_dvb_xy.x, drgb_dvb_xy.y, scalar(0.0), drgb_dw.y + dd_dwy};
    vec4 drgb_dvc = {drgb_dvc_xy.x, drgb_dvc_xy.y, scalar(0.0), drgb_dw.z + dd_dwz};

    atomicAdd4(&grad_vertices[tri.a], drgb_dva);
    atomicAdd4(&grad_vertices[tri.b], drgb_dvb);
    atomicAdd4(&grad_vertices[tri.c], drgb_dvc);
}

__global__ void render_forward_kernel(
    color4 *__restrict__ image, scalar *__restrict__ depthmap, scalar *__restrict__ final_weights, int *__restrict__ ends,
    const scalar *bary_transforms,
    const vec4 *__restrict__ vertices, const id3 *__restrict__ indices, const color3 *__restrict__ colors, const scalar *__restrict__ opacities,
    const int *__restrict__ per_tile_list, const int *__restrict__ offsets,
    int width, int height,
    const scalar early_stopping_threshold,
    bool per_pixel_sort,
    int max_layers)
{
    int tile_index = blockIdx.y * gridDim.x + blockIdx.x;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (iy >= height || ix >= width)
        return;

    int index = iy * width + ix;
    int start = offsets[tile_index];
    int end = offsets[tile_index + 1];

    vec2 pos = {((scalar)ix + 0.5) / (scalar)width * 2.0 - 1.0,
                ((scalar)iy + 0.5) / (scalar)height * 2.0 - 1.0};

    scalar alpha = scalar(1.0);
    color3 total_color = {scalar(0.0), scalar(0.0), scalar(0.0)};
    scalar total_depth = scalar(0.0);
    scalar total_weight = scalar(0.0);
    int end_out = 0;

    constexpr const int layers = 32;
    int batch_j[layers];
    scalar batch_depths[layers];

    int blended_count = 0;

    if (!per_pixel_sort)
    {
        for (int j = start; j < end && alpha > early_stopping_threshold && blended_count < max_layers; ++j)
        {
            process_forward(j, per_tile_list, bary_transforms, vertices, indices, colors, opacities, pos, alpha, total_color, total_depth, total_weight, end_out, blended_count);
        }
    }
    else
    {
        scalar lastDepth = -std::numeric_limits<scalar>::infinity();
        int lastJ = -1;

        while (alpha > early_stopping_threshold && blended_count < max_layers)
        {
            int batch_size = std::min(layers, max_layers - blended_count);
            int found = select_next_k_in_tile(batch_size, batch_j, batch_depths, lastDepth, lastJ, start, end,
                                              per_tile_list, bary_transforms, vertices, indices, pos);

            if (found == 0)
                break;

            int to_process = std::min(found, max_layers - blended_count);

            for (int i = 0; i < to_process; ++i)
            {
                int j = batch_j[i];
                process_forward(j, per_tile_list, bary_transforms, vertices, indices, colors, opacities, pos, alpha, total_color, total_depth, total_weight, end_out, blended_count);
            }

            lastDepth = batch_depths[to_process - 1];
            lastJ = batch_j[to_process - 1];
        }
    }
    if (total_weight > eff_zero)
        depthmap[index] = total_depth / total_weight;
    else
    {
        depthmap[index] = std::numeric_limits<scalar>::infinity();
        total_weight = scalar(0.0);
    }
    final_weights[index] = total_weight;
    ends[index] = end_out;
    image[index] = color4(total_color, alpha);
}

__global__ void render_backward_kernel(
    vec4 *__restrict__ grad_vertices, color3 *__restrict__ grad_colors, scalar *__restrict__ grad_opacities,
    const color4 *__restrict__ grad_output, const scalar *__restrict__ grad_depthmap,
    const color4 *__restrict__ image, const scalar *__restrict__ depthmap, const scalar *__restrict__ final_weights, const int *__restrict__ ends,
    const int *__restrict__ sorted_ids, const int *__restrict__ offsets,
    const scalar *__restrict__ bary_transforms,
    const vec4 *__restrict__ vertices, const id3 *__restrict__ indices, const color3 *__restrict__ colors, const scalar *__restrict__ opacities,
    int width, int height,
    const scalar early_stopping_threshold,
    bool per_pixel_sort,
    int max_layers)
{
    int tile_index = blockIdx.y * gridDim.x + blockIdx.x;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (iy >= height || ix >= width)
        return;

    int index = iy * width + ix;
    int start = offsets[tile_index];
    int tile_end = offsets[tile_index + 1];
    int processed_limit = std::min(ends[index], max_layers);

    vec2 pos = {((scalar)ix + 0.5) / (scalar)width * 2.0 - 1.0,
                ((scalar)iy + 0.5) / (scalar)height * 2.0 - 1.0};

    scalar final_alpha = image[index].a;
    scalar alpha = final_alpha;
    vec3 s = {scalar(0.0), scalar(0.0), scalar(0.0)};
    scalar final_weight = final_weights[index];
    scalar final_depth = depthmap[index];
    scalar oa = scalar(0.0);
    scalar zoa = scalar(0.0);

    constexpr const int layers = 32;
    int batch_j[layers];
    scalar batch_depths[layers];

    if (!per_pixel_sort)
    {
        for (int i = 0; i < processed_limit; ++i)
        {
            int j = start + i;
            process_backward(j, sorted_ids, bary_transforms, vertices, indices, colors, opacities, pos, grad_output, grad_depthmap, index, final_alpha, alpha, s, grad_vertices, grad_colors, grad_opacities, oa, zoa, final_weight, final_depth);
        }
    }
    else
    {
        scalar lastDepth = -std::numeric_limits<scalar>::infinity();
        int lastJ = -1;
        int remaining = processed_limit;

        while (remaining > 0)
        {
            int batch_size = min(layers, remaining);
            int found = select_next_k_in_tile(batch_size, batch_j, batch_depths, lastDepth, lastJ, start, tile_end,
                                              sorted_ids, bary_transforms, vertices, indices, pos);

            if (found == 0)
                break;

            int to_process = min(found, remaining);

            for (int idx = to_process - 1; idx >= 0; --idx)
            {
                int k = batch_j[idx];
                process_backward(k, sorted_ids, bary_transforms, vertices, indices, colors, opacities, pos, grad_output, grad_depthmap, index, final_alpha, alpha, s, grad_vertices, grad_colors, grad_opacities, oa, zoa, final_weight, final_depth);
            }

            lastDepth = batch_depths[to_process - 1];
            lastJ = batch_j[to_process - 1];
            remaining -= to_process;
        }
    }
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
        const_vec4(vertices),
        const_id3(indices),
        triangle_count);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::map<std::string, float>> render_forward(
    torch::Tensor vertices, torch::Tensor indices, torch::Tensor colors, torch::Tensor opacities, // input: vertices, indices, colors and opacities
    int width, int height,                                                                        // image size
    const int tile_width, const int tile_height,                                                  // tile size
    const scalar early_stopping_threshold,                                                        // remaining opacity at which to stop rendering
    bool disable_timing, bool per_pixel_sort, int max_layers)
{
    std::map<std::string, float> timings;
    CudaTimer timer(timings, disable_timing);

    timer.start("init");
    vertices = vertices.contiguous();
    indices = indices.contiguous();
    colors = colors.contiguous();
    opacities = opacities.contiguous();
    auto image = torch::zeros({height, width, 4}, torch::TensorOptions(torchscalar).device(torch::kCUDA));
    auto depthmap = torch::zeros({height, width}, torch::TensorOptions(torchscalar).device(torch::kCUDA));
    auto final_weights = torch::zeros({height, width}, torch::TensorOptions(torchscalar).device(torch::kCUDA));
    auto ends = torch::zeros({height, width}, torch::TensorOptions(torch::kInt32).device(torch::kCUDA));

    timer.start("per_tile_lists");
    auto [ids, per_tile_depths, offsets] = per_tile_lists(vertices, indices, width, height, tile_width, tile_height);

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
        mutable_color4(image),
        mutable_scalar(depthmap),
        mutable_scalar(final_weights),
        mutable_int(ends),
        const_scalar(bary_transforms),
        const_vec4(vertices),
        const_id3(indices), const_color3(colors), const_scalar(opacities),
        const_int(ids), const_int(offsets),
        width, height,
        early_stopping_threshold, per_pixel_sort, max_layers);
    timer.stop();
    return {image, depthmap, final_weights, ids, offsets, bary_transforms, ends, timings};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> render_backward(
    torch::Tensor grad_output, torch::Tensor grad_depthmap,                                       // upstream gradient
    torch::Tensor vertices, torch::Tensor indices, torch::Tensor colors, torch::Tensor opacities, // triangle data
    torch::Tensor sorted_ids, torch::Tensor offsets,                                              // ids and offsets for sorted rendering
    torch::Tensor bary_transforms,                                                                // cartesian to barycentric transformation matrices
    torch::Tensor image, torch::Tensor depthmap, torch::Tensor final_weights, torch::Tensor ends, // final opacities and indices
    int width, int height,                                                                        // image size
    const int tile_width, const int tile_height,                                                  // tile size
    const scalar early_stopping_threshold,                                                        // remaining opacity at which to stop rendering
    bool per_pixel_sort, int max_layers)
{
    colors = colors.contiguous();
    vertices = vertices.contiguous();
    indices = indices.contiguous();
    opacities = opacities.contiguous();
    bary_transforms = bary_transforms.contiguous();
    sorted_ids = sorted_ids.contiguous();
    offsets = offsets.contiguous();
    image = image.contiguous();
    depthmap = depthmap.contiguous();
    final_weights = final_weights.contiguous();
    grad_depthmap = grad_depthmap.contiguous();
    grad_output = grad_output.contiguous();
    ends = ends.contiguous();
    auto grad_vertices = torch::zeros_like(vertices);
    auto grad_colors = torch::zeros_like(colors);
    auto grad_opacities = torch::zeros_like(opacities);
    const dim3 threads_per_block(tile_width, tile_height);
    const dim3 blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);
    render_backward_kernel<<<blocks, threads_per_block>>>(
        mutable_vec4(grad_vertices), mutable_color3(grad_colors), mutable_scalar(grad_opacities),
        const_color4(grad_output), const_scalar(grad_depthmap),
        const_color4(image), const_scalar(depthmap), const_scalar(final_weights), const_int(ends),
        const_int(sorted_ids), const_int(offsets),
        const_scalar(bary_transforms),
        const_vec4(vertices), const_id3(indices), const_color3(colors), const_scalar(opacities),
        width, height,
        early_stopping_threshold, per_pixel_sort, max_layers);
    return {grad_vertices, grad_colors, grad_opacities};
}
