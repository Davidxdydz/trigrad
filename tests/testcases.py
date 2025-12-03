import numpy as np
import torch


def add_w(vertices):
    N = vertices.shape[0]
    w = torch.ones((N, 1), device=vertices.device, dtype=vertices.dtype)
    return torch.cat([vertices, w], dim=1)


def overlapping_squares(N, mins=0.1, maxs=0.9):
    vertices = []
    indices = []
    colors = []
    sizes = torch.linspace(mins, maxs, N)
    for i, s in enumerate(sizes):
        s = s / 2
        f = 1.00001
        vertices.append([-f * s, -s, i])
        vertices.append([f * s, -s, i])
        vertices.append([f * s, s, i])
        vertices.append([-f * s, s, i])
        a, b, c, d = i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3
        indices.append([a, b, c])
        indices.append([a, c, d])
        c = [torch.rand((1,)).item() for _ in range(3)]
        colors.append(c)
        colors.append(c)
        colors.append(c)
        colors.append(c)
    vertices = torch.tensor(vertices) * 2
    indices = torch.tensor(indices, dtype=torch.int32)
    colors = torch.tensor(colors)
    vertices = add_w(vertices)
    return vertices, indices, colors


def depth_overlap():
    vertices = torch.tensor(
        [
            [-1.0, -1, 0, 1],
            [1, -1, 0, 1],
            [0, 1, -1, 1],
            [-1, 1, 0, 1],
            [1, 1, 0, 1],
            [0, -1, -1, 1],
        ],
    )
    indices = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=torch.int32,
    )
    colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
    )
    opacities = torch.ones(vertices.shape[0])
    return vertices, indices, colors, opacities


def large_trianlges(N, r=0.5, unicolor=False):
    vertices = []
    indices = []
    colors = []
    for i in range(N):
        angle = (torch.rand((1,)) * 2 * torch.pi).item()
        vertices.append([np.cos(angle) * r, np.sin(angle) * r, i])
        vertices.append([np.cos(angle + 2 * np.pi / 3) * r, np.sin(angle + 2 * np.pi / 3) * r, i])
        vertices.append(
            [
                np.cos(angle + 2 * 2 * np.pi / 3) * r,
                np.sin(angle + 2 * 2 * np.pi / 3) * r,
                i,
            ]
        )
        indices.append([i * 3, i * 3 + 1, i * 3 + 2])
        c = [torch.rand((1,)).item() for _ in range(3)]
        colors.append(c)
        if not unicolor:
            c = [torch.rand((1,)).item() for _ in range(3)]
        colors.append(c)
        if not unicolor:
            c = [torch.rand((1,)).item() for _ in range(3)]
        colors.append(c)
    vertices = torch.tensor(vertices)
    vertices[:, :2] += 0.5
    indices = torch.tensor(indices, dtype=torch.int32)
    colors = torch.tensor(colors)
    vertices = add_w(vertices)
    return vertices, indices, colors


def overlapping_triangles():
    vertices = torch.tensor(
        [
            [0.1, 0.1, 0],
            [0.1, 0.9, 0],
            [0.75, 0.5, 0],
            [0.9, 0.1, 1],
            [0.9, 0.9, 1],
            [0.25, 0.5, 1],
        ],
    )
    indices = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=torch.int32,
    )
    colors = torch.tensor(
        [
            [1.0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
    )
    vertices = add_w(vertices) * 2 - 1
    return vertices, indices, colors


def test_square():
    vertices = torch.tensor(
        [
            [0.1, 0.1, 0],
            [0.1, 0.9, 0],
            [0.9, 0.1, 0],
            [0.9, 0.9, 0],
        ],
    )
    indices = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.int32,
    )

    colors = torch.tensor(
        [
            [1.0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
    )
    vertices = add_w(vertices) * 2 - 1
    return vertices, indices, colors


def grid_mesh(nx, ny, minx=-1, maxx=1, miny=-1, maxy=1):
    vertices = []
    indices = []
    for iy in range(ny):
        y = miny + iy * (maxy - miny) / (ny - 1)
        for ix in range(nx):
            step_x = (maxx - minx) / (nx - 1)
            if iy % 2 == 0:
                x = minx + ix * step_x
            else:
                if ix == nx - 1:
                    break
                x = minx + step_x / 2 + ix * step_x
            vertices.append([x, y, 0])
            n = len(vertices) - 1
            # up and right triangle
            if iy < ny - 1 and ((ix < nx - 2) or (ix < nx - 1 and iy % 2 == 0)) and iy < ny - 1:
                indices.append([n, n + 1, n + nx])
            # up center triangle
            if iy < ny - 1 and (iy % 2 == 1 or (ix > 0 and ix < nx - 1)):
                indices.append([n, n + nx, n + nx - 1])
    vertices = torch.tensor(vertices)
    indices = torch.tensor(indices, dtype=torch.int32)
    colors = torch.rand((vertices.shape[0], 3))
    vertices = add_w(vertices)
    return vertices, indices, colors


def overlap_mesh():
    vertices = torch.tensor(
        [
            [-0.5, 0, 0],
            [0.5, 0, 0],
            [-0.2, 1.2, 0],
            [-0.0, 0.3, 0],
            [-0.5, 1, 0],
            [0.5, 1, 0],
            [0, 1.5, 0],
            [0, -0.5, 0],
            [1, -0.5, 0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [0, 3, 2],
            [1, 2, 3],
            [0, 1, 3],
            [0, 2, 4],
            [1, 5, 2],
            [5, 6, 2],
            [4, 2, 6],
            [7, 1, 0],
            [8, 1, 7],
            [8, 5, 1],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    colors = torch.rand((vertices.shape[0], 3), device="cuda", dtype=torch.float32)
    vertices = add_w(vertices)
    return vertices, indices, colors
