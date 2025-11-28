from typing import Tuple
import torch
from torch.autograd import Function
from torch.autograd.function import FunctionCtx
import trigrad._C as _C
from trigrad.util import create_check

check_tensor = create_check()


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
    ) -> torch.Tensor:
        N, _ = vertices.shape
        check_tensor(vertices, "vertices", (N, 4))
        check_tensor(colors, "colors", (N, 3))
        check_tensor(opacities, "opacities", (N,))
        N, _ = indices.shape
        check_tensor(indices, "indices", (N, 3), torch.int32)

        ctx.width = width
        ctx.height = height
        ctx.tile_width = tile_width
        ctx.tile_height = tile_height
        ctx.early_stopping_threshold = early_stopping_threshold
        ctx.record_timing = record_timing
        if N < 1:
            ctx.skipped = True
            image = torch.zeros((height, width, 3), dtype=colors.dtype, device=colors.device)
            timings = None
            ctx.save_for_backward(vertices, indices, colors, opacities)
        else:
            ctx.skipped = False
            image, *args, timings = _C.render_forward(
                vertices,
                indices,
                colors,
                opacities,
                vertices[:, 2][indices].mean(dim=-1),  # depths
                width,
                height,
                tile_width,
                tile_height,
                early_stopping_threshold,
                not record_timing,
            )
            ctx.save_for_backward(vertices, indices, colors, opacities, *args)

        if record_timing:
            return image, timings
        return image

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        if ctx.record_timing:
            grad_output, grad_timings = grad_output
        else:
            grad_output = grad_output[0]
        if ctx.skipped:
            grad_vertices = torch.zeros_like(ctx.saved_tensors[0])
            grad_colors = torch.zeros_like(ctx.saved_tensors[2])
            grad_opacities = torch.zeros_like(ctx.saved_tensors[3])
        else:
            grad_vertices, grad_colors, grad_opacities = _C.render_backward(
                grad_output,
                *ctx.saved_tensors,
                ctx.width,
                ctx.height,
                ctx.tile_width,
                ctx.tile_height,
                ctx.early_stopping_threshold,
            )
        return (
            grad_vertices,  # vertices
            None,  # indices
            grad_colors,  # colors
            grad_opacities,  # opacities
            # None,  # depths
            None,  # width
            None,  # height
            None,  # tile_width
            None,  # tile_height
            None,  # early_stopping_threshold
            None,  # record_timing
        )


def render(
    vertices: torch.Tensor,
    indices: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    width: int = 500,
    height: int = 500,
    tile_width=16,
    tile_height=16,
    early_stopping_threshold=1 / 256,
    record_timing=False,
) -> torch.Tensor:
    """
    vertices : (N, 4) tensor of triangle vertices (x'/w',y'/w',z'/w',1/w')
    indices : (M, 3) tensor of triangle vertex indices
    colors : (N, 3) tensor of triangle vertex colors
    opacities : (N,) tensor of triangle vertex opacities

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
    )
