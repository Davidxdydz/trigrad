from tests.testcases import overlapping_squares, large_trianlges
import torch
from torch.autograd import gradcheck
import trigrad
from pytest import mark

tile_size = 4
torch.set_default_device("cpu")
torch.set_default_dtype(trigrad.precision)


def uniform(shape, min=-5, max=5, requires_grad=True):
    t = torch.rand(shape) * (max - min) + min
    if requires_grad:
        t.requires_grad_()
    return t


def uniforms(shape, n, min=-5, max=5, requires_grad=True):
    return [uniform(shape, min, max, requires_grad) for _ in range(n)]


@mark.parametrize("bary", [uniform(3, 0, 1, requires_grad=False) for _ in range(10)])
@mark.parametrize("scalars", [uniforms(1, 3) for _ in range(10)])
@mark.parametrize("weights", [uniform(3, 0, 5) for _ in range(10)])
def test_interpolate3_scalar(bary, scalars, weights):
    bary = bary / bary.sum()
    bary.requires_grad_()
    a, b, c = scalars
    assert gradcheck(trigrad.interpolate3_scalar, (bary, a, b, c, weights))


@mark.parametrize("bary", [uniform(3, 0, 1, requires_grad=False) for _ in range(10)])
@mark.parametrize("vectors", [uniforms(3, 3) for _ in range(10)])
@mark.parametrize("weights", [uniform(3, 0, 5) for _ in range(10)])
def test_interpolate3_vector(bary, vectors, weights):
    bary = bary / bary.sum()
    bary.requires_grad_()
    a, b, c = vectors
    assert gradcheck(trigrad.interpolate3_vector, (bary, a, b, c, weights))


@mark.parametrize("points", [uniforms(2, 4) for _ in range(100)])
def test_barycentric(points):
    p0, p1, q0, q1 = points
    assert gradcheck(trigrad.barycentric, (p0, p1, q0, q1))
