from typing import Tuple
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
import trigrad._C as _C
from trigrad.util import create_check

check_tensor = create_check("cuda", _C.precision)
precision = _C.precision


class Renderer(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        width: int = 500,
        height: int = 500,
        tile_width: int = 16,
        tile_height: int = 16,
        early_stopping_threshold: float = 1 / 256,
        record_timing: bool = False,
        per_pixel_sort: bool = True,
        max_layers: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
        N, _ = vertices.shape
        check_tensor(vertices, "vertices", (N, 4))
        check_tensor(colors, "colors", (N, 3))
        check_tensor(opacities, "opacities", (N,))
        N, _ = indices.shape
        check_tensor(indices, "indices", (N, 3), torch.int32)
        if max_layers <= 0:
            max_layers = N + 1

        ctx.width = width
        ctx.height = height
        ctx.tile_width = tile_width
        ctx.tile_height = tile_height
        ctx.early_stopping_threshold = early_stopping_threshold
        ctx.record_timing = record_timing
        ctx.per_pixel_sort = per_pixel_sort
        ctx.max_layers = max_layers
        if N < 1:
            # TODO handle this case correctly
            ctx.skipped = True
            image = torch.zeros((height, width, 4), dtype=colors.dtype, device=colors.device)
            image[..., 3] = 1.0
            depthmap = torch.ones((height, width), dtype=colors.dtype, device=colors.device) * float("inf")
            timings = None
            ctx.save_for_backward(vertices, indices, colors, opacities)
        else:
            ctx.skipped = False
            image, depthmap, final_weights, sorted_ids, offsets, bary_transforms, ends, timings = _C.render_forward(
                vertices,
                indices,
                colors,
                opacities,
                width,
                height,
                tile_width,
                tile_height,
                early_stopping_threshold,
                not record_timing,
                per_pixel_sort,
                max_layers,
            )
            ctx.save_for_backward(vertices, indices, colors, opacities, sorted_ids, offsets, bary_transforms, image, depthmap, final_weights, ends)
        if record_timing:
            return image, depthmap, timings
        return image, depthmap

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_image: torch.Tensor,
        grad_depthmap: torch.Tensor,
        grad_timings=None,
    ):
        if ctx.skipped:
            grad_vertices = torch.zeros_like(ctx.saved_tensors[0])
            grad_colors = torch.zeros_like(ctx.saved_tensors[2])
            grad_opacities = torch.zeros_like(ctx.saved_tensors[3])
        else:
            grad_vertices, grad_colors, grad_opacities = _C.render_backward(
                grad_image,
                grad_depthmap,
                *ctx.saved_tensors,
                ctx.width,
                ctx.height,
                ctx.tile_width,
                ctx.tile_height,
                ctx.early_stopping_threshold,
                ctx.per_pixel_sort,
                ctx.max_layers,
            )
        return (
            grad_vertices,  # vertices
            None,  # indices
            grad_colors,  # colors
            grad_opacities,  # opacities
            None,  # width
            None,  # height
            None,  # tile_width
            None,  # tile_height
            None,  # early_stopping_threshold
            None,  # record_timing
            None,  # per_pixel_sort
            None,  # max_layers
        )


def render(
    vertices: torch.Tensor,
    indices: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    width: int = 500,
    height: int = 500,
    tile_width: int = 16,
    tile_height: int = 16,
    early_stopping_threshold: float = 1 / 256,
    record_timing: bool = False,
    per_pixel_sort: bool = True,
    max_layers: int = 32,
) -> torch.Tensor | Tuple[torch.Tensor, dict]:
    """
    Render semi-transparent triangles using a software rasterizer.
        vertices : (N, 4) tensor of triangle vertices (x'/w',y'/w',z'/w',1/w') in [-1,1] clip space
        indices : (M, 3) tensor of triangle vertex indices
        colors : (N, 3) tensor of triangle vertex colors
        opacities : (N,) tensor of triangle vertex opacities
        per_pixel_sort : bool indicating whether to sort fragments per pixel or per tile
        max_layers : maximum number of layers to composite per pixel, <= 0 for no limit

    Returns:
        image : (height, width, 4) tensor of rendered image **The alpha channel contains translucency, not opacity, for numerical stability**
        depthmap : (height, width) tensor of depth values
        timings (optional) : dict of timing information, only if record_timing is True
    """
    if tile_width * tile_height > 1024:
        raise ValueError("tile_width * tile_height must be <= 1024")
    return Renderer.apply(
        vertices,
        indices,
        colors,
        opacities,
        width,
        height,
        tile_width,
        tile_height,
        early_stopping_threshold,
        record_timing,
        per_pixel_sort,
        max_layers,
    )
