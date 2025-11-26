import numpy as np
import torch


def overlapping_squares(N, mins=0.1, maxs=0.9):
    vertices = []
    indices = []
    colors = []
    depths = []
    sizes = torch.linspace(mins, maxs, N)
    for i, s in enumerate(sizes):
        s = s / 2
        f = 1.00001
        vertices.append([-f * s, -s])
        vertices.append([f * s, -s])
        vertices.append([f * s, s])
        vertices.append([-f * s, s])
        a, b, c, d = i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3
        indices.append([a, b, c])
        indices.append([a, c, d])
        depths.extend([float(i), float(i)])
        c = [torch.rand((1,)).item() for _ in range(3)]
        colors.append(c)
        colors.append(c)
        colors.append(c)
        colors.append(c)
    vertices = torch.tensor(vertices)
    vertices += 0.5
    indices = torch.tensor(indices, dtype=torch.int32)
    colors = torch.tensor(colors)
    depths = torch.tensor(depths)
    return vertices, indices, colors, depths


def large_trianlges(N, r=0.5, unicolor=False):
    vertices = []
    indices = []
    colors = []
    depths = []
    for i in range(N):
        angle = (torch.rand((1,)) * 2 * torch.pi).item()
        vertices.append([np.cos(angle) * r, np.sin(angle) * r])
        vertices.append([np.cos(angle + 2 * np.pi / 3) * r, np.sin(angle + 2 * np.pi / 3) * r])
        vertices.append(
            [
                np.cos(angle + 2 * 2 * np.pi / 3) * r,
                np.sin(angle + 2 * 2 * np.pi / 3) * r,
            ]
        )
        indices.append([i * 3, i * 3 + 1, i * 3 + 2])
        depths.append(float(i))
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
    depths = torch.tensor(depths)
    return vertices, indices, colors, depths


def overlapping_triangles():
    vertices = torch.tensor(
        [
            [0.1, 0.1],
            [0.1, 0.9],
            [0.75, 0.5],
            [0.9, 0.1],
            [0.9, 0.9],
            [0.25, 0.5],
        ],
    )
    indices = torch.tensor(
        [
            [0.0, 1, 2],
            [3, 4, 5],
        ],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [0.0, 1.0],
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
    return vertices, indices, colors, depths


def test_square():
    vertices = torch.tensor(
        [
            [0.1, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.9],
        ],
    )
    indices = torch.tensor(
        [[0, 1, 2], [1, 2, 3]],
        dtype=torch.int32,
    )
    depths = torch.tensor(
        [0.0, 0],
    )
    colors = torch.tensor(
        [
            [1.0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
    )
    return vertices, indices, colors, depths


def grid_mesh(nx, ny, minx=0, maxx=1, miny=0, maxy=1):
    vertices = []
    indices = []
    depths = []
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
            vertices.append([x, y])
            n = len(vertices) - 1
            # up and right triangle
            if iy < ny - 1 and ((ix < nx - 2) or (ix < nx - 1 and iy % 2 == 0)) and iy < ny - 1:
                indices.append([n, n + 1, n + nx])
                depths.append(0.0)
            # up center triangle
            if iy < ny - 1 and (iy % 2 == 1 or (ix > 0 and ix < nx - 1)):
                indices.append([n, n + nx, n + nx - 1])
                depths.append(0.0)
    vertices = torch.tensor(vertices)
    indices = torch.tensor(indices, dtype=torch.int32)
    colors = torch.rand((vertices.shape[0], 3))
    depths = torch.tensor(depths)
    return vertices, indices, colors, depths


def overlap_mesh():
    vertices = torch.tensor(
        [
            [-0.5, 0],
            [0.5, 0],
            [-0.2, 1.2],
            [-0.0, 0.3],
            [-0.5, 1],
            [0.5, 1],
            [0, 1.5],
            [0, -0.5],
            [1, -0.5],
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
    return vertices, indices, colors
