#include "util.h"

const vec4 *const_vec4(torch::Tensor tensor)
{
    return reinterpret_cast<const vec4 *>(tensor.const_data_ptr<scalar>());
}
vec4 *mutable_vec4(torch::Tensor tensor)
{
    return reinterpret_cast<vec4 *>(tensor.mutable_data_ptr<scalar>());
}
const vec3 *const_vec3(torch::Tensor tensor)
{
    return reinterpret_cast<const vec3 *>(tensor.const_data_ptr<scalar>());
}

vec2 *mutable_vec2(torch::Tensor tensor)
{
    return reinterpret_cast<vec2 *>(tensor.mutable_data_ptr<scalar>());
}

const vec2 *const_vec2(torch::Tensor tensor)
{
    return reinterpret_cast<const vec2 *>(tensor.const_data_ptr<scalar>());
}

vec3 *mutable_vec3(torch::Tensor tensor)
{
    return reinterpret_cast<vec3 *>(tensor.mutable_data_ptr<scalar>());
}

const color3 *const_color3(torch::Tensor tensor)
{
    return reinterpret_cast<const color3 *>(tensor.const_data_ptr<scalar>());
}
color3 *mutable_color3(torch::Tensor tensor)
{
    return reinterpret_cast<color3 *>(tensor.mutable_data_ptr<scalar>());
}
const id3 *const_id3(torch::Tensor tensor)
{
    return reinterpret_cast<const id3 *>(tensor.const_data_ptr<int>());
}
id3 *mutable_id3(torch::Tensor tensor)
{
    return reinterpret_cast<id3 *>(tensor.mutable_data_ptr<int>());
}

int *mutable_int(torch::Tensor tensor)
{
    return tensor.mutable_data_ptr<int>();
}

scalar *mutable_scalar(torch::Tensor tensor)
{
    return tensor.mutable_data_ptr<scalar>();
}

const scalar *const_scalar(torch::Tensor tensor)
{
    return tensor.const_data_ptr<scalar>();
}
const int *const_int(torch::Tensor tensor)
{
    return tensor.const_data_ptr<int>();
}

const id4 *const_id4(torch::Tensor tensor)
{
    return reinterpret_cast<const id4 *>(tensor.const_data_ptr<int>());
}

bool *mutable_bool(torch::Tensor tensor)
{
    return tensor.mutable_data_ptr<bool>();
}

const bool *const_bool(torch::Tensor tensor)
{
    return tensor.const_data_ptr<bool>();
}

const color4 *const_color4(torch::Tensor tensor)
{
    return reinterpret_cast<const color4 *>(tensor.const_data_ptr<scalar>());
}
color4 *mutable_color4(torch::Tensor tensor)
{
    return reinterpret_cast<color4 *>(tensor.mutable_data_ptr<scalar>());
}