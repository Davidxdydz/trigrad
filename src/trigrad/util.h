
#pragma once
#include <torch/extension.h>
#include <cuda_runtime.h>

// #define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
using vec4 = double4;
using vec3 = double3;
using vec2 = double2;
using scalar = double;
inline constexpr auto torchscalar = torch::kFloat64;
#else
using vec4 = float4;
using vec3 = float3;
using vec2 = float2;
using scalar = float;
inline constexpr auto torchscalar = torch::kFloat32;
#endif

static __device__ inline constexpr scalar max_opacity = 0.9999;

constexpr scalar eff_zero = std::numeric_limits<scalar>::min();

__host__ __device__ inline vec3 operator*(const vec3 &a, const vec3 &b)
{

    vec3 result = {a.x * b.x, a.y * b.y, a.z * b.z};

    return result;
}

__host__ __device__ inline vec3 operator*(const vec3 &a, const scalar &b)
{

    vec3 result = {a.x * b, a.y * b, a.z * b};

    return result;
}

__host__ __device__ inline vec4 operator*(const vec4 &a, const scalar &b)
{

    vec4 result = {a.x * b, a.y * b, a.z * b, a.w * b};

    return result;
}

__host__ __device__ inline vec4 operator*(const vec4 &a, const vec4 &b)
{

    vec4 result = {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};

    return result;
}

__host__ __device__ inline vec3 operator*(const scalar &a, const vec3 &b)
{

    vec3 result = {a * b.x, a * b.y, a * b.z};

    return result;
}

__host__ __device__ inline vec3 operator+(const vec3 &a, const vec3 &b)
{

    vec3 result = {a.x + b.x, a.y + b.y, a.z + b.z};

    return result;
}

__host__ __device__ inline vec3 operator-(const vec3 &a, const vec3 &b)
{

    vec3 result = {a.x - b.x, a.y - b.y, a.z - b.z};

    return result;
}

__host__ __device__ inline vec3 operator/(const vec3 &a, const scalar &b)
{

    return {a.x / b, a.y / b, a.z / b};
}

__host__ __device__ inline vec3 operator/(const vec3 &a, const vec3 &b)
{

    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__host__ __device__ inline vec2 operator-(const vec2 &a)
{

    return {a.x, a.y};
}

__host__ __device__ inline vec2 operator+(const vec2 &a, const scalar &b)
{

    return {a.x + b, a.y + b};
}

__host__ __device__ inline vec2 operator/(const vec2 &a, const vec2 &b)
{

    return {a.x / b.x, a.y / b.y};
}

__host__ __device__ inline vec2 operator-(const vec2 &a, const vec2 &b)
{

    return {a.x - b.x, a.y - b.y};
}

__host__ __device__ inline vec2 operator*(const scalar &a, const vec2 &b)
{

    return {a * b.x, a * b.y};
}
__host__ __device__ inline vec2 operator+(const vec2 &a, const vec2 &b)
{

    return {a.x + b.x, a.y + b.y};
}

__host__ __device__ inline vec3 operator+(const vec3 &a, const scalar &b)
{

    return {a.x + b, a.y + b, a.z + b};
}

__host__ __device__ inline vec3 operator+=(vec3 &a, const vec3 &b)
{

    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline vec3 operator/(const scalar &a, const vec3 &b)
{

    return {a / b.x, a / b.y, a / b.z};
}

__host__ __device__ inline vec2 operator*(const vec2 &a, const scalar &b)
{

    return {a.x * b, a.y * b};
}

__host__ __device__ inline vec2 operator/(const vec2 &a, const scalar &b)
{

    return {a.x / b, a.y / b};
}

__host__ __device__ inline vec2 operator-(const vec2 &a, const scalar &b)
{

    return {a.x - b, a.y - b};
}

__host__ __device__ inline scalar component_sum(const vec3 &a)
{

    return a.x + a.y + a.z;
}

__host__ __device__ inline vec3 operator-(const vec3 &a)
{

    return {-a.x, -a.y, -a.z};
}

__host__ __device__ inline vec2 xy(const vec3 &a)
{

    return {a.x, a.y};
}
__host__ __device__ inline vec2 xy(const vec4 &a)
{

    return {a.x, a.y};
}
__host__ __device__ inline vec3 xyz(const vec4 &a)
{

    return {a.x, a.y, a.z};
}

__host__ __device__ inline vec3 interpolate3(vec3 bary, vec3 a, vec3 b, vec3 c, vec3 w = {1, 1, 1})
{

    scalar denom = bary.x * w.x + bary.y * w.y + bary.z * w.z;
    return (bary.x * a * w.x + bary.y * b * w.y + bary.z * c * w.z) / denom;
}

__host__ __device__ inline scalar interpolate3(vec3 bary, scalar a, scalar b, scalar c, vec3 w = {1, 1, 1})
{

    scalar denom = bary.x * w.x + bary.y * w.y + bary.z * w.z;
    return (bary.x * a * w.x + bary.y * b * w.y + bary.z * c * w.z) / denom;
}

__host__ __device__ inline scalar lerp(scalar a, scalar b, scalar t, vec2 w = {1, 1})
{

    scalar denom = (1 - t) * w.x + t * w.y;
    return (a * (1 - t) * w.x + b * t * w.y) / denom;
}

__host__ __device__ inline vec3 lerp(vec3 a, vec3 b, scalar t, vec2 w = {1, 1})
{

    scalar denom = (1 - t) * w.x + t * w.y;
    return (a * (1 - t) * w.x + b * t * w.y) / denom;
}

__host__ __device__ inline vec2 lerp(vec2 a, vec2 b, scalar t, vec2 w = {1, 1})
{

    scalar denom = (1 - t) * w.x + t * w.y;
    return (a * (1 - t) * w.x + b * t * w.y) / denom;
}

__host__ __device__ inline vec3 cross(vec3 a, vec3 b)
{

    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

__host__ __device__ inline scalar dot(const vec2 &a, const vec2 &b)
{

    return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline scalar dot(const vec3 &a, const vec3 &b)
{

    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline scalar dot(const vec4 &a, const vec4 &b)
{

    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__host__ __device__ inline std::tuple<vec3, scalar, scalar, scalar, vec3>
interpolate3_backward(scalar d_dinter, vec3 bary, scalar a, scalar b, scalar c, const vec3 w)
{

    vec3 vals = {a, b, c};

    scalar nom = component_sum(bary * w * vals);
    scalar denom = component_sum(bary * w);
    if (denom * denom < eff_zero)
        return {{0, 0, 0}, 0, 0, 0, {0, 0, 0}};
    scalar inv_denom2 = 1 / (denom * denom);

    vec3 dfd = (vals * w * denom - nom * w) * inv_denom2;
    vec3 grad_bary = d_dinter * dfd;
    vec3 grad_vals = d_dinter * bary * w / denom;
    vec3 grad_w = d_dinter * (bary * vals * denom - nom * bary) * inv_denom2;

    return {grad_bary, grad_vals.x, grad_vals.y, grad_vals.z, grad_w};
}

__host__ __device__ inline std::tuple<vec3, vec3, vec3, vec3, vec3>
interpolate3_backward(vec3 d_dinter, vec3 bary, vec3 a, vec3 b, vec3 c, vec3 w)
{

    vec3 nom = bary.x * w.x * a + bary.y * w.y * b + bary.z * w.z * c;
    scalar denom = bary.x * w.x + bary.y * w.y + bary.z * w.z;
    if (denom * denom < eff_zero)
        return {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    scalar inv_denom2 = 1.0 / (denom * denom);

    vec3 df_dbx = w.x * (a * denom - nom) * inv_denom2;
    vec3 df_dby = w.y * (b * denom - nom) * inv_denom2;
    vec3 df_dbz = w.z * (c * denom - nom) * inv_denom2;

    vec3 grad_bary = {
        dot(d_dinter, df_dbx),
        dot(d_dinter, df_dby),
        dot(d_dinter, df_dbz),
    };

    vec3 scale = bary * w / denom;

    vec3 grad_a = d_dinter * scale.x;
    vec3 grad_b = d_dinter * scale.y;
    vec3 grad_c = d_dinter * scale.z;

    vec3 df_dw0 = (bary.x * a * denom - nom * bary.x) * inv_denom2;
    vec3 df_dw1 = (bary.y * b * denom - nom * bary.y) * inv_denom2;
    vec3 df_dw2 = (bary.z * c * denom - nom * bary.z) * inv_denom2;

    vec3 grad_w = {
        dot(d_dinter, df_dw0),
        dot(d_dinter, df_dw1),
        dot(d_dinter, df_dw2)};

    return {grad_bary, grad_a, grad_b, grad_c, grad_w};
}

__host__ __device__ inline vec3 mean(const vec3 &a, const vec3 &b, const vec3 &c)
{

    return (a + b + c) / scalar(3.0);
}

__device__ inline vec3 normalize(const vec3 &a)
{

    scalar len = std::sqrt(dot(a, a));
    if (len > 0)
        return a / len;
    else
        return {0, 0, 0};
}

__host__ __device__ inline scalar dist(vec3 a, vec3 b)
{

    scalar d = dot(a - b, a - b);
    if (d < eff_zero)

        return 0.0;

    return std::sqrt(d);
}

__host__ __device__ inline scalar cross2d(vec2 a, vec2 b)
{

    return a.x * b.y - a.y * b.x;
}

__device__ inline vec4 to4(const vec2 &a, scalar z = 0.0, scalar w = 0.0)
{
    return {a.x, a.y, z, w};
}

__host__ __device__ inline vec3 barycentric(vec2 v0, vec2 v1, vec2 v2, vec2 p)
{

    scalar T = cross2d(v1 - v0, v2 - v0);
    scalar a = cross2d(v1 - p, v2 - p) / T;
    scalar b = cross2d(v2 - p, v0 - p) / T;
    scalar c = cross2d(v0 - p, v1 - p) / T;
    return {a, b, c};
}

__host__ __device__ inline std::tuple<vec2, vec2, vec2, vec2> barycentric_backward(vec3 d_db, vec2 a, vec2 b, vec2 c, vec2 p)
{

    vec2 dbx_da, dby_da, dbz_da; // d bary_0 / d a, d bary_1 / d a, d bary_2 / d a
    vec2 dbx_db, dby_db, dbz_db; // d bary_0 / d b, d bary_1 / d b, d bary_2 / d b
    vec2 dbx_dc, dby_dc, dbz_dc; // d bary_0 / d c, d bary_1 / d c, d bary_2 / d c
    vec2 dbx_dp, dby_dp, dbz_dp; // d bary_0 / d p, d bary_1 / d p, d bary_2 / d p

    vec3 l = barycentric(a, b, c, p);

    scalar d = a.x * (b.y - c.y) + b.x * (-a.y + c.y) + c.x * (a.y - b.y);
    if (std::abs(d) < eff_zero)
    {
        return {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    }
    dbx_da.x = l.x * (-b.y + c.y) / d;
    dby_da.x = (-c.y - l.y * (b.y - c.y) + p.y) / d;
    dbz_da.x = (b.y - l.z * (b.y - c.y) - p.y) / d;
    dbx_da.y = l.x * (b.x - c.x) / d;
    dby_da.y = (c.x + l.y * (b.x - c.x) - p.x) / d;
    dbz_da.y = (-b.x + l.z * (b.x - c.x) + p.x) / d;
    dbx_db.x = (c.y + l.x * (a.y - c.y) - p.y) / d;
    dby_db.x = l.y * (a.y - c.y) / d;
    dbz_db.x = (-a.y + l.z * (a.y - c.y) + p.y) / d;
    dbx_db.y = (-c.x - l.x * (a.x - c.x) + p.x) / d;
    dby_db.y = l.y * (-a.x + c.x) / d;
    dbz_db.y = (a.x - l.z * (a.x - c.x) - p.x) / d;
    dbx_dc.x = (-b.y - l.x * (a.y - b.y) + p.y) / d;
    dby_dc.x = (a.y - l.y * (a.y - b.y) - p.y) / d;
    dbz_dc.x = l.z * (-a.y + b.y) / d;
    dbx_dc.y = (b.x + l.x * (a.x - b.x) - p.x) / d;
    dby_dc.y = (-a.x + l.y * (a.x - b.x) + p.x) / d;
    dbz_dc.y = l.z * (a.x - b.x) / d;
    dbx_dp.x = (b.y - c.y) / d;
    dby_dp.x = (-a.y + c.y) / d;
    dbz_dp.x = (a.y - b.y) / d;
    dbx_dp.y = (-b.x + c.x) / d;
    dby_dp.y = (a.x - c.x) / d;
    dbz_dp.y = (-a.x + b.x) / d;

    vec2 da = {d_db.x * dbx_da.x + d_db.y * dby_da.x + d_db.z * dbz_da.x,
               d_db.x * dbx_da.y + d_db.y * dby_da.y + d_db.z * dbz_da.y};
    vec2 db = {d_db.x * dbx_db.x + d_db.y * dby_db.x + d_db.z * dbz_db.x,
               d_db.x * dbx_db.y + d_db.y * dby_db.y + d_db.z * dbz_db.y};
    vec2 dc = {d_db.x * dbx_dc.x + d_db.y * dby_dc.x + d_db.z * dbz_dc.x,
               d_db.x * dbx_dc.y + d_db.y * dby_dc.y + d_db.z * dbz_dc.y};
    vec2 dp = {d_db.x * dbx_dp.x + d_db.y * dby_dp.x + d_db.z * dbz_dp.x,
               d_db.x * dbx_dp.y + d_db.y * dby_dp.y + d_db.z * dbz_dp.y};

    return {da, db, dc, dp};
}

union color3
{
    vec3 rgb;
    struct
    {
        scalar r, g, b;
    };

    // Default constructor
    __host__ __device__ color3() : rgb{0.0, 0.0, 0.0} {}

    // Construct from individual components
    __host__ __device__ color3(scalar red, scalar green, scalar blue)
    {
        rgb = {red, green, blue};
    }

    // Construct from CUDA vec3
    __host__ __device__ color3(const vec3 &v)
    {
        rgb = v;
    }

    // Assignment from vec3
    __host__ __device__ color3 &operator=(const vec3 &v)
    {
        rgb = v;
        return *this;
    }

    // Implicit conversion to vec3
    __host__ __device__ operator vec3() const
    {
        return rgb;
    }
};

union color4
{
    vec4 rgba;
    struct
    {
        scalar r, g, b, a;
    };

    // Default constructor
    __host__ __device__ color4() : rgba{0.0, 0.0, 0.0, 0.0} {}
    // Construct from individual components
    __host__ __device__ color4(scalar red, scalar green, scalar blue, scalar alpha)
    {
        rgba = {red, green, blue, alpha};
    }

    // Construct from CUDA vec4
    __host__ __device__ color4(const vec4 &v)
    {
        rgba = v;
    }

    // Assignment from vec4
    __host__ __device__ color4 &operator=(const vec4 &v)
    {
        rgba = v;
        return *this;
    }

    // Implicit conversion to vec4
    __host__ __device__ operator vec4() const
    {
        return rgba;
    }
    __host__ __device__ color4(const color3 &c, scalar alpha)
    {
        rgba = {c.r, c.g, c.b, alpha};
    }

    __host__ __device__ inline color3 rgb() const
    {
        return {r, g, b};
    }
};

union id3
{
    int3 abc;
    struct
    {
        int a, b, c;
    };

    // Default constructor
    __host__ __device__ id3() : abc{0, 0, 0} {}

    // Construct from individual components
    __host__ __device__ constexpr id3(int aa, int bb, int cc) : abc{aa, bb, cc}
    {
    }

    // Construct from CUDA int3
    __host__ __device__ id3(const int3 &v)
    {
        abc = v;
    }

    // Assignment from int3
    __host__ __device__ id3 &operator=(const int3 &v)
    {
        abc = v;
        return *this;
    }

    // Implicit conversion to int3
    __host__ __device__ operator int3() const
    {
        return abc;
    }
};

union id4
{
    int4 abcd;
    struct
    {
        int a, b, c, d;
    };

    // Default constructor
    __host__ __device__ id4() : abcd{0, 0, 0, 0} {}

    // Construct from individual components
    __host__ __device__ id4(int aa, int bb, int cc, int dd)
    {
        abcd = make_int4(aa, bb, cc, dd);
    }

    // Construct from CUDA int3
    __host__ __device__ id4(const int4 &v)
    {
        abcd = v;
    }

    // Assignment from int3
    __host__ __device__ id4 &operator=(const int4 &v)
    {
        abcd = v;
        return *this;
    }

    // Implicit conversion to int3
    __host__ __device__ operator int4() const
    {
        return abcd;
    }
};

union id2
{
    int2 ab;
    struct
    {
        int a, b;
    };

    // Default constructor
    __host__ __device__ id2() : ab{0, 0} {}

    // Construct from individual components
    __host__ __device__ id2(int aa, int bb)
    {
        ab = make_int2(aa, bb);
    }

    // Construct from CUDA int3
    __host__ __device__ id2(const int2 &v)
    {
        ab = v;
    }

    // Assignment from int3
    __host__ __device__ id2 &operator=(const int2 &v)
    {
        ab = v;
        return *this;
    }

    // Implicit conversion to int3
    __host__ __device__ operator int2() const
    {
        return ab;
    }
};

template <typename T>
__device__ inline void swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;
}

__device__ inline id3 select(const id4 &from, const id3 &ids)
{
    int *from_ids = (int *)&from;
    return {from_ids[ids.a], from_ids[ids.b], from_ids[ids.c]};
}
__device__ inline int select(const id4 &from, const int &id)
{
    int *from_ids = (int *)&from;
    return from_ids[id];
}

__device__ inline id2 select(const id3 &from, const id2 &ids)
{
    int *from_ids = (int *)&from;
    return {from_ids[ids.a], from_ids[ids.b]};
}

__device__ inline id2 select(const id4 &from, const id2 &ids)
{
    int *from_ids = (int *)&from;
    return {from_ids[ids.a], from_ids[ids.b]};
}

__device__ inline void atomicAdd3(vec3 *address, vec3 value)
{
    atomicAdd(&address->x, value.x);
    atomicAdd(&address->y, value.y);
    atomicAdd(&address->z, value.z);
}

__device__ inline void atomicAdd3(color3 *address, color3 value)
{
    atomicAdd(&address->r, value.r);
    atomicAdd(&address->g, value.g);
    atomicAdd(&address->b, value.b);
}

__device__ inline void atomicAdd4(vec4 *address, vec4 value)
{
    atomicAdd(&address->x, value.x);
    atomicAdd(&address->y, value.y);
    atomicAdd(&address->z, value.z);
    atomicAdd(&address->w, value.w);
}

const vec3 *const_vec3(torch::Tensor tensor);
const vec2 *const_vec2(torch::Tensor tensor);
vec2 *mutable_vec2(torch::Tensor tensor);
vec3 *mutable_vec3(torch::Tensor tensor);
const color3 *const_color3(torch::Tensor tensor);
color3 *mutable_color3(torch::Tensor tensor);
const id3 *const_id3(torch::Tensor tensor);
id3 *mutable_id3(torch::Tensor tensor);
int *mutable_int(torch::Tensor tensor);
scalar *mutable_scalar(torch::Tensor tensor);
const scalar *const_scalar(torch::Tensor tensor);
const int *const_int(torch::Tensor tensor);
const id4 *const_id4(torch::Tensor tensor);
const vec4 *const_vec4(torch::Tensor tensor);
vec4 *mutable_vec4(torch::Tensor tensor);
const color4 *const_color4(torch::Tensor tensor);
color4 *mutable_color4(torch::Tensor tensor);
bool *mutable_bool(torch::Tensor tensor);
const bool *const_bool(torch::Tensor tensor);