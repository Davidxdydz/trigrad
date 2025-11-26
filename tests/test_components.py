import torch
import difftet
from torch.autograd import gradcheck
from pytest import mark


def uniform(shape, min=-5, max=5, dtype=difftet.precision, requires_grad=True):
    t = torch.rand(shape, dtype=dtype) * (max - min) + min
    if requires_grad:
        t.requires_grad_()
    return t


def uniforms(shape, n, min=-5, max=5, dtype=difftet.precision, requires_grad=True):
    return [uniform(shape, min, max, dtype, requires_grad) for _ in range(n)]


def random_points_2d():
    return [uniform(2) for _ in range(4)]


@mark.parametrize("points", [uniforms(2, 4) for _ in range(100)])
def test_intersection(points):
    p0, p1, q0, q1 = points
    assert gradcheck(difftet.intersection, (p0, p1, q0, q1))


@mark.parametrize("v", [uniform(3) for _ in range(10)])
@mark.parametrize(
    "projection_matrix",
    [uniform((4, 4), -1, 1, requires_grad=False) for _ in range(10)],
)
def test_project(v, projection_matrix):
    assert gradcheck(difftet.project, (v, projection_matrix))


@mark.parametrize("points", [random_points_2d() for _ in range(100)])
def test_bary(points):
    p0, p1, q0, q1 = points
    assert gradcheck(difftet.bary, (p0, p1, q0, q1))


@mark.parametrize("colors", [uniforms(3, 2) for _ in range(10)])
@mark.parametrize("opacities", [uniforms(1, 2) for _ in range(10)])
@mark.parametrize("distance", [uniform(1, 0, 50) for _ in range(10)])
def test_compute_color(opacities, colors, distance):
    o0, o1 = opacities
    c0, c1 = colors
    d = distance
    assert gradcheck(difftet.compute_color, (o0, o1, c0, c1, d))


@mark.parametrize("opacities", [uniforms(1, 2) for _ in range(10)])
@mark.parametrize("depth", [uniform(1, 0, 50) for _ in range(10)])
@mark.parametrize("merge_depth_opacity", [True, False])
def test_compute_opacity(opacities, depth, merge_depth_opacity):
    o0, o1 = opacities
    d = depth
    assert gradcheck(difftet.compute_opacity, (o0, o1, d, merge_depth_opacity))


@mark.parametrize("positions", [uniforms(3, 2) for _ in range(100)])
def test_dist(positions):
    p0, p1 = positions
    assert gradcheck(difftet.dist, (p0, p1), check_grad_dtypes=True)


@mark.parametrize("bary", [uniform(3, 0, 1, requires_grad=False) for _ in range(10)])
@mark.parametrize("scalars", [uniforms(1, 3) for _ in range(10)])
@mark.parametrize("weights", [uniform(3) for _ in range(10)])
def test_interpolate3_scalar(bary, scalars, weights):
    bary = bary / bary.sum()
    bary.requires_grad_()
    a, b, c = scalars
    assert gradcheck(difftet.interpolate3_scalar, (bary, a, b, c, weights))


@mark.parametrize("bary", [uniform(3, 0, 1, requires_grad=False) for _ in range(10)])
@mark.parametrize("vectors", [uniforms(3, 3) for _ in range(10)])
@mark.parametrize("weights", [uniform(3) for _ in range(10)])
def test_interpolate3_vector(bary, vectors, weights):
    bary = bary / bary.sum()
    bary.requires_grad_()
    a, b, c = vectors
    assert gradcheck(difftet.interpolate3_vector, (bary, a, b, c, weights))


@mark.parametrize("t", [uniform(1, 0, 1) for _ in range(10)])
@mark.parametrize("scalars", [uniforms(1, 2) for _ in range(10)])
@mark.parametrize("weights", [uniform(2) for _ in range(10)])
def test_lerp_scalar(t, scalars, weights):
    t.requires_grad_()
    a, b = scalars
    assert gradcheck(difftet.lerp_scalar, (a, b, t, weights))


@mark.parametrize("t", [uniform(1, 0, 1) for _ in range(10)])
@mark.parametrize("vectors", [uniforms(3, 2) for _ in range(10)])
@mark.parametrize("weights", [uniform(2) for _ in range(10)])
def test_lerp_vector(t, vectors, weights):
    t.requires_grad_()
    a, b = vectors
    assert gradcheck(difftet.lerp_vector, (a, b, t, weights))
