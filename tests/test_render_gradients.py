from tests.testcases import overlapping_squares, large_trianlges
import torch
from torch.autograd import gradcheck
import trigrad
import pytest

tile_size = 4
torch.set_default_device("cuda")
torch.set_default_dtype(trigrad.precision)


def test_dc():
    vertices, indices, colors = overlapping_squares(10)
    opacities = torch.full_like(vertices[:, 0], 0.99)
    colors.requires_grad_()
    assert gradcheck(
        trigrad.render,
        (
            vertices,
            indices,
            colors,
            opacities,
            50,
            50,
            tile_size,
            tile_size,
            1 / 256,
        ),
    )


def test_do():
    vertices, indices, colors = overlapping_squares(10)
    opacities = torch.full_like(vertices[:, 0], 0.99)
    opacities.requires_grad_()
    assert gradcheck(
        trigrad.render,
        (
            vertices,
            indices,
            colors,
            opacities,
            50,
            50,
            tile_size,
            tile_size,
            1 / 256,
        ),
    )


def test_dv():
    vertices, indices, colors = large_trianlges(3, r=4, unicolor=False)
    opacities = torch.full_like(vertices[:, 0], 0.99)
    vertices.requires_grad_()
    assert gradcheck(
        trigrad.render,
        (
            vertices,
            indices,
            colors,
            opacities,
            50,
            50,
            tile_size,
            tile_size,
            1 / 256,
        ),
    )


def test_dall():
    vertices, indices, colors = large_trianlges(3, r=4, unicolor=False)
    opacities = torch.full_like(vertices[:, 0], 0.99)
    vertices.requires_grad_()
    colors.requires_grad_()
    opacities.requires_grad_()
    assert gradcheck(
        trigrad.render,
        (
            vertices,
            indices,
            colors,
            opacities,
            50,
            50,
            tile_size,
            tile_size,
            1 / 256,
        ),
    )
