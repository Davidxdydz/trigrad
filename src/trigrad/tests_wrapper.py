from typing import Tuple
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
import trigrad._C as _C
from trigrad.util import create_check

check_tensor = create_check("cpu")


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
