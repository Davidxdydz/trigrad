from tests.testcases import overlapping_squares, large_trianlges
import torch
from torch.autograd import gradcheck
import difftet
import pytest

tile_size = 4
torch.set_default_device("cuda")
torch.set_default_dtype(difftet.precision)


@pytest.mark.parametrize("opacity", [0.9, 1.0])
def test_dc(opacity):
    vertices, indices, colors = overlapping_squares(10)
    opacities = torch.full_like(vertices[:, 0], opacity)
    colors.requires_grad_()
    assert gradcheck(
        difftet.render,
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


@pytest.mark.parametrize("opacity", [0.9, 1.0])
def test_do(opacity):
    vertices, indices, colors = overlapping_squares(10)

    opacities = torch.full_like(vertices[:, 0], opacity)
    opacities.requires_grad_()
    assert gradcheck(
        difftet.render,
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


@pytest.mark.parametrize("opacity", [0.9, 1.0])
def test_dv(opacity):
    vertices, indices, colors = large_trianlges(3, r=4, unicolor=False)
    opacities = torch.full_like(vertices[:, 0], opacity)
    vertices.requires_grad_()
    assert gradcheck(
        difftet.render,
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


@pytest.mark.parametrize("opacity", [0.9, 1.0])
def test_dall(opacity):
    vertices, indices, colors = large_trianlges(3, r=4, unicolor=False)
    opacities = torch.full_like(vertices[:, 0], opacity)
    vertices.requires_grad_()
    colors.requires_grad_()
    opacities.requires_grad_()
    assert gradcheck(
        difftet.render,
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
