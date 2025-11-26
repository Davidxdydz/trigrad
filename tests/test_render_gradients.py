from tests.testcases import overlapping_squares, large_trianlges
import torch
from torch.autograd import gradcheck
import difftet
import pytest

tile_size = 4
torch.set_default_device("cuda")
torch.set_default_dtype(difftet.precision)


@pytest.mark.parametrize(
    "activate_opacity",
    [False, True],
    ids=["opacity_interpolation", "depth_interpolation"],
)
def test_dc(activate_opacity):
    vertices, indices, colors, depths = overlapping_squares(10)

    opacities = torch.ones(vertices.shape[0]) * 0.9
    if activate_opacity:
        opacities = torch.ones(vertices.shape[0]) * 2
    colors.requires_grad_()
    assert gradcheck(
        difftet.render,
        (
            vertices,
            indices,
            colors,
            opacities,
            depths,
            50,
            50,
            tile_size,
            tile_size,
            1 / 256,
            activate_opacity,
        ),
    )


@pytest.mark.parametrize(
    "activate_opacity",
    [False, True],
    ids=["opacity_interpolation", "depth_interpolation"],
)
def test_do(activate_opacity):
    vertices, indices, colors, depths = overlapping_squares(10)

    opacities = torch.ones(vertices.shape[0]) * 0.9
    if activate_opacity:
        opacities = torch.ones(vertices.shape[0]) * 2
    opacities.requires_grad_()
    assert gradcheck(
        difftet.render,
        (
            vertices,
            indices,
            colors,
            opacities,
            depths,
            50,
            50,
            tile_size,
            tile_size,
            1 / 256,
            activate_opacity,
        ),
    )


@pytest.mark.parametrize(
    "activate_opacity",
    [False, True],
    ids=["opacity_interpolation", "depth_interpolation"],
)
def test_dv(activate_opacity):
    vertices, indices, colors, depths = large_trianlges(3, r=4, unicolor=False)
    opacities = torch.ones(vertices.shape[0]) * 0.5
    if activate_opacity:
        opacities = torch.ones(vertices.shape[0]) * 2
    vertices.requires_grad_()
    assert gradcheck(
        difftet.render,
        (
            vertices,
            indices,
            colors,
            opacities,
            depths,
            50,
            50,
            tile_size,
            tile_size,
            1 / 256,
            activate_opacity,
        ),
    )


@pytest.mark.parametrize(
    "activate_opacity",
    [False, True],
    ids=["opacity_interpolation", "depth_interpolation"],
)
def test_dall(activate_opacity):
    vertices, indices, colors, depths = large_trianlges(3, r=4, unicolor=False)
    opacities = torch.ones(vertices.shape[0]) * 0.5
    if activate_opacity:
        opacities = torch.ones(vertices.shape[0]) * 2
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
            depths,
            50,
            50,
            tile_size,
            tile_size,
            1 / 256,
            activate_opacity,
        ),
    )
