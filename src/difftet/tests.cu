#include "tests.h"
#include "util.h"
// #include "render.cu"

vec2 to_vec2(torch::Tensor tensor)
{
    return {tensor[0].item<scalar>(), tensor[1].item<scalar>()};
}

vec3 to_vec3(torch::Tensor tensor)
{
    return {tensor[0].item<scalar>(), tensor[1].item<scalar>(), tensor[2].item<scalar>()};
}

vec4 to_vec4(torch::Tensor tensor)
{
    return {tensor[0].item<scalar>(), tensor[1].item<scalar>(), tensor[2].item<scalar>(), tensor[3].item<scalar>()};
}
scalar to_scalar(torch::Tensor tensor)
{
    return tensor.item<scalar>();
}

torch::Tensor to_tensor(vec3 d)
{
    return torch::tensor({d.x, d.y, d.z}, torch::dtype(torchscalar));
}

torch::Tensor to_tensor(vec4 d)
{
    return torch::tensor({d.x, d.y, d.z, d.w}, torch::dtype(torchscalar));
}

torch::Tensor to_tensor(vec2 d)
{
    return torch::tensor({d.x, d.y}, torch::dtype(torchscalar));
}

torch::Tensor to_tensor(scalar d)
{
    return torch::tensor({d}, torch::dtype(torchscalar));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> bary_backward_torch(torch::Tensor d_db, torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor p)
{
    auto [da, db, dc, dp] = bary_backward(to_vec3(d_db), to_vec2(a), to_vec2(b), to_vec2(c), to_vec2(p));
    return {to_tensor(da), to_tensor(db), to_tensor(dc), to_tensor(dp)};
}

torch::Tensor bary_torch(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor p)
{
    auto result = bary(to_vec2(a), to_vec2(b), to_vec2(c), to_vec2(p));
    return to_tensor(result);
}