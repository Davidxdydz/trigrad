from typing import Tuple
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
import difftet._C as _C
from difftet.util import create_check

check_tensor = create_check("cpu")


# class Intersection(Function):
#     @staticmethod
#     def forward(ctx: FunctionCtx, p0, p1, q0, q1):
#         check_tensor(p0, "p0", 2)
#         check_tensor(p1, "p1", 2)
#         check_tensor(q0, "q0", 2)
#         check_tensor(q1, "q1", 2)

#         ctx.save_for_backward(p0, p1, q0, q1)
#         return _C.intersection(p0, p1, q0, q1)

#     @staticmethod
#     def backward(ctx: FunctionCtx, grad_t0: torch.Tensor, grad_t1: torch.Tensor):
#         grad_p0, grad_p1, grad_q0, grad_q1 = _C.intersection_backward(
#             grad_t0, grad_t1, *ctx.saved_tensors
#         )
#         return grad_p0, grad_p1, grad_q0, grad_q1


# def intersection(p0, p1, q0, q1):
#     return Intersection.apply(p0, p1, q0, q1)


# class Project(Function):
#     @staticmethod
#     def forward(ctx: FunctionCtx, v, projection_matrix):
#         check_tensor(v, "v", 3)
#         check_tensor(projection_matrix, "projection_matrix", (4, 4))
#         ctx.save_for_backward(v, projection_matrix)
#         return _C.project(v, projection_matrix)

#     @staticmethod
#     def backward(ctx: FunctionCtx, grad_proj: torch.Tensor):
#         grad_v = _C.project_backward(grad_proj, *ctx.saved_tensors)
#         return grad_v, None


# def project(v, projection_matrix):
#     return Project.apply(v, projection_matrix)


class Barycentric(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, p0, p1, q0, q1):
        check_tensor(p0, "p0", 2)
        check_tensor(p1, "p1", 2)
        check_tensor(q0, "q0", 2)
        check_tensor(q1, "q1", 2)

        ctx.save_for_backward(p0, p1, q0, q1)
        return _C.barycentric_torch(p0, p1, q0, q1)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_bary: torch.Tensor):
        grad_p0, grad_p1, grad_q0, grad_q1 = _C.barycentric_backward_torch(grad_bary, *ctx.saved_tensors)
        return grad_p0, grad_p1, grad_q0, grad_q1


def barycentric(p0, p1, q0, q1):
    return Barycentric.apply(p0, p1, q0, q1)


# class ComputeColor(Function):
#     @staticmethod
#     def forward(ctx: FunctionCtx, o0, o1, c0, c1, d):
#         check_tensor(o0, "o0", 1)
#         check_tensor(o1, "o1", 1)
#         check_tensor(c0, "c0", 3)
#         check_tensor(c1, "c1", 3)
#         check_tensor(d, "d", 1)

#         ctx.save_for_backward(o0, o1, c0, c1, d)
#         return _C.compute_color(o0, o1, c0, c1, d)

#     @staticmethod
#     def backward(ctx: FunctionCtx, grad_color: torch.Tensor):
#         grad_o0, grad_o1, grad_c0, grad_c1, grad_d = _C.compute_color_backward(
#             grad_color, *ctx.saved_tensors
#         )
#         return grad_o0, grad_o1, grad_c0, grad_c1, grad_d


# def compute_color(o0, o1, c0, c1, d):
#     return ComputeColor.apply(o0, o1, c0, c1, d)


# class ComputeOpacity(Function):
#     @staticmethod
#     def forward(ctx: FunctionCtx, o0, o1, d, merge_depth_opacity):
#         check_tensor(o0, "o0", 1)
#         check_tensor(o1, "o1", 1)
#         check_tensor(d, "d", 1)

#         ctx.save_for_backward(o0, o1, d)
#         ctx.merge_depth_opacity = merge_depth_opacity
#         return _C.compute_opacity(o0, o1, d, merge_depth_opacity)

#     @staticmethod
#     def backward(ctx: FunctionCtx, grad_opacity: torch.Tensor):
#         grad_o0, grad_o1, grad_d = _C.compute_opacity_backward(
#             grad_opacity, *ctx.saved_tensors, ctx.merge_depth_opacity
#         )
#         return grad_o0, grad_o1, grad_d, None


# def compute_opacity(o0, o1, d, merge_depth_opacity):
#     return ComputeOpacity.apply(o0, o1, d, merge_depth_opacity)


# class Dist(Function):
#     @staticmethod
#     def forward(ctx: FunctionCtx, p0, p1):
#         check_tensor(p0, "p0", 3)
#         check_tensor(p1, "p1", 3)

#         ctx.save_for_backward(p0, p1)
#         return _C.dist(p0, p1)

#     @staticmethod
#     def backward(ctx: FunctionCtx, grad_dist: torch.Tensor):
#         grad_p0, grad_p1 = _C.dist_backward(grad_dist, *ctx.saved_tensors)
#         return grad_p0, grad_p1


# def dist(p0, p1):
#     return Dist.apply(p0, p1)


class Interpolate3Scalar(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        bary: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        w: torch.Tensor,
    ):
        check_tensor(bary, "bary", 3)
        check_tensor(a, "a", 1)
        check_tensor(b, "b", 1)
        check_tensor(c, "c", 1)
        check_tensor(w, "w", 3)

        ctx.save_for_backward(bary, a, b, c, w)
        return _C.interpolate3_scalar_torch(bary, a, b, c, w)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: torch.Tensor):
        grad_bary, grad_a, grad_b, grad_c, grad_w = _C.interpolate3_scalar_backward_torch(grad_out, *ctx.saved_tensors)
        return grad_bary, grad_a, grad_b, grad_c, grad_w


def interpolate3_scalar(bary, a, b, c, w):
    return Interpolate3Scalar.apply(bary, a, b, c, w)


class Interpolate3Vector(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        bary: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        w: torch.Tensor,
    ):
        check_tensor(bary, "bary", 3)
        check_tensor(a, "a", 3)
        check_tensor(b, "b", 3)
        check_tensor(c, "c", 3)
        check_tensor(w, "w", 3)

        ctx.save_for_backward(bary, a, b, c, w)
        return _C.interpolate3_vector_torch(bary, a, b, c, w)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: torch.Tensor):
        grad_bary, grad_a, grad_b, grad_c, grad_w = _C.interpolate3_vector_backward_torch(grad_out, *ctx.saved_tensors)
        return grad_bary, grad_a, grad_b, grad_c, grad_w


def interpolate3_vector(bary, a, b, c, w):
    return Interpolate3Vector.apply(bary, a, b, c, w)


# class LerpScalar(Function):
#     @staticmethod
#     def forward(
#         ctx: FunctionCtx,
#         a: torch.Tensor,
#         b: torch.Tensor,
#         t: torch.Tensor,
#         w: torch.Tensor,
#     ):
#         check_tensor(a, "a", 1)
#         check_tensor(b, "b", 1)
#         check_tensor(t, "t", 1)
#         check_tensor(w, "w", 2)

#         ctx.save_for_backward(a, b, t, w)
#         return _C.lerp_scalar(a, b, t, w)

#     @staticmethod
#     def backward(ctx: FunctionCtx, grad_out: torch.Tensor):
#         grad_a, grad_b, grad_t, grad_w = _C.lerp_scalar_backward(
#             grad_out, *ctx.saved_tensors
#         )
#         return grad_a, grad_b, grad_t, grad_w


# def lerp_scalar(a, b, t, w):
#     return LerpScalar.apply(a, b, t, w)


# class LerpVector(Function):
#     @staticmethod
#     def forward(
#         ctx: FunctionCtx,
#         a: torch.Tensor,
#         b: torch.Tensor,
#         t: torch.Tensor,
#         w: torch.Tensor,
#     ):
#         check_tensor(a, "a", 3)
#         check_tensor(b, "b", 3)
#         check_tensor(t, "t", 1)
#         check_tensor(w, "w", 2)

#         ctx.save_for_backward(a, b, t, w)
#         return _C.lerp_vector(a, b, t, w)

#     @staticmethod
#     def backward(ctx: FunctionCtx, grad_out: torch.Tensor):
#         grad_a, grad_b, grad_t, grad_w = _C.lerp_vector_backward(
#             grad_out, *ctx.saved_tensors
#         )
#         return grad_a, grad_b, grad_t, grad_w


# def lerp_vector(a, b, t, w):
#     return LerpVector.apply(a, b, t, w)
