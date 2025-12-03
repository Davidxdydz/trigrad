import torch
import numpy as np
import trigrad
import argparse
import trimesh
from pathlib import Path


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class Camera:
    def __init__(
        self,
        position=[0, 0, 0],
        fov=30,
        near=0.01,
        far=1000,
        width=500,
        height=500,
        view=None,
        dtype=np.float64,
    ):
        self.__fov = fov
        self.__near = near
        self.__far = far
        self.__width = width
        self.__height = height
        self.position = np.array(position, dtype=np.float64)
        self.dtype = dtype
        self.P = self.perspective()
        if view is None:
            view = np.eye(4, dtype=dtype)
        self.V = view

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, value):
        self.__width = value
        self.P = self.perspective()

    @property
    def height(self):
        return self.__height

    @height.setter
    def height(self, value):
        self.__height = value
        self.P = self.perspective()

    @property
    def fov(self):
        return self.__fov

    @fov.setter
    def fov(self, value):
        self.__fov = value
        self.P = self.perspective()

    @property
    def near(self):
        return self.__near

    @near.setter
    def near(self, value):
        self.__near = value
        self.P = self.perspective()

    @property
    def far(self):
        return self.__far

    @far.setter
    def far(self, value):
        self.__far = value
        self.P = self.perspective()

    @property
    def aspect_ratio(self):
        if self.height == 0:
            return 1.0
        return self.width / self.height

    @property
    def forward(self):
        forward = np.linalg.inv(self.V)[2, :3]
        return normalize(forward)

    def perspective(self):
        f = self.far
        n = self.near

        t = np.tan(np.radians(self.fov) / 2) * n
        r = t * self.width / self.height

        return np.array(
            [
                [n / r, 0, 0, 0],
                [0, n / t, 0, 0],
                [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
                [0, 0, -1, 0],
            ],
            dtype=self.dtype,
        )

    @property
    def PV(self):
        return self.P @ self.V

    def __repr__(self):
        return f"Camera(view={self.V}, projection={self.P}, position={self.position})"

    @property
    def PV_cuda(self):
        return torch.tensor(self.PV, dtype=trigrad.precision, device="cuda")

    @property
    def position_cuda(self):
        return torch.tensor(self.position, dtype=trigrad.precision, device="cuda")

    def lookat(self, target: np.ndarray | list, up: np.ndarray | list = [0, 1, 0]):
        if not isinstance(target, np.ndarray):
            target = np.array(target, dtype=self.dtype)
        eye = self.position
        up = normalize(up)
        forward = normalize(eye - target)
        right = normalize(np.cross(up, forward))
        up = np.cross(forward, right)
        return np.array(
            [
                [right[0], right[1], right[2], -np.dot(right, eye)],
                [up[0], up[1], up[2], -np.dot(up, eye)],
                [forward[0], forward[1], forward[2], -np.dot(forward, eye)],
                [0, 0, 0, 1],
            ],
            dtype=self.dtype,
        )


class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def normalize(self):
        norm = np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm == 0:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def to_rotation_matrix(self):
        return np.array(
            [
                [
                    1 - 2 * (self.y**2 + self.z**2),
                    2 * (self.x * self.y - self.w * self.z),
                    2 * (self.x * self.z + self.w * self.y),
                ],
                [
                    2 * (self.x * self.y + self.w * self.z),
                    1 - 2 * (self.x**2 + self.z**2),
                    2 * (self.y * self.z - self.w * self.x),
                ],
                [
                    2 * (self.x * self.z - self.w * self.y),
                    2 * (self.y * self.z + self.w * self.x),
                    1 - 2 * (self.x**2 + self.y**2),
                ],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def axis_angle(axis, angle):
        axis = np.array(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        w = np.cos(half_angle)
        x = axis[0] * np.sin(half_angle)
        y = axis[1] * np.sin(half_angle)
        z = axis[2] * np.sin(half_angle)
        return Quaternion(w, x, y, z).normalize()


class SphericalCamera(Camera):
    def __init__(self, distance=2, fov=30):
        super().__init__(fov=fov)
        self.__distance = distance
        self.rotation = np.eye(3, dtype=self.dtype)
        self.update()

    @property
    def distance(self):
        return self.__distance

    @distance.setter
    def distance(self, value):
        self.__distance = value
        self.update()

    def mouse_x(self, dx):
        axis = self.rotation[:, 1]  # Y-axis
        quaternion = Quaternion.axis_angle(axis, -dx)
        rotation_matrix = quaternion.to_rotation_matrix()
        self.rotation = rotation_matrix @ self.rotation
        self.update()

    def mouse_y(self, dy):
        axis = self.rotation[:3, 0]
        quaternion = Quaternion.axis_angle(axis, dy)
        rotation_matrix = quaternion.to_rotation_matrix()
        self.rotation = rotation_matrix @ self.rotation
        self.update()

    def update(self):
        self.position = np.array(
            [
                0,
                0,
                np.exp(self.distance),
            ],
            dtype=self.dtype,
        )
        self.position = self.rotation[:3, :3] @ self.position
        self.V = self.lookat([0, 0, 0], self.rotation[:3, 1])


import torch
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
from PyQt6.QtGui import QMouseEvent, QWheelEvent
import sys


from PyQt6 import QtCore, QtWidgets

from dataclasses import dataclass
import trigrad


@dataclass
class Mesh:
    vertices: torch.Tensor
    indices: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    name: str | None = None


class Display(QtWidgets.QLabel):
    def __init__(self, mesh: Mesh, camera=None):
        super().__init__()
        if camera is None:
            camera = SphericalCamera()
        self.camera = camera
        self.mesh = mesh
        self.frame_time = 0.0
        self.frame_time_ema = 0.9
        # Allow the pixmap to scale arbitrarily.
        self.setScaledContents(True)

    def render_frame(self):
        self.camera.width = self.width()
        self.camera.height = self.height()
        homogeneous = torch.ones((self.mesh.vertices.shape[0], 1), dtype=self.mesh.vertices.dtype, device=self.mesh.vertices.device)
        vertices_h = torch.cat([self.mesh.vertices[:, :3], homogeneous], dim=-1)
        vertices_cam = (self.camera.PV_cuda @ vertices_h.T).T
        vertices_ndc = torch.zeros_like(vertices_cam)
        vertices_ndc[:, :3] = vertices_cam[:, :3] / vertices_cam[:, 3:4]
        vertices_ndc[:, 3] = 1 / vertices_cam[:, 3]
        vertices_final = torch.cat([vertices_ndc, self.mesh.vertices[:, 3:4]], dim=-1)
        frame, timings = trigrad.render(vertices_final, self.mesh.indices, self.mesh.colors, self.mesh.opacities, width=self.camera.width, height=self.camera.height, record_timing=True)
        total_time = sum(timings.values())
        self.frame_time = self.frame_time * self.frame_time_ema + total_time * (1 - self.frame_time_ema)
        print(f"Frame time: {self.frame_time*1000:.2f} ms, {1/self.frame_time:.2f} FPS")
        frame = frame.detach().cpu().numpy().clip(0, 1)

        frame = (frame * 255).astype(np.uint8)
        return frame

    def update_image(self):
        frame_np = self.render_frame()

        height, width, channels = frame_np.shape
        bytes_per_line = channels * width
        qimg = QtGui.QImage(
            frame_np.data,
            width,
            height,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))


class Viewer(QtWidgets.QWidget):
    def __init__(self, mesh: Mesh, dt=0, camera=None):
        super().__init__()

        self.display = Display(mesh, camera)

        self.mesh = mesh

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.display)
        self.setLayout(layout)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.display.update_image)
        self.timer.start(int(dt * 1000))
        self.resize(camera.width, camera.height)
        self.setMinimumSize(300, 300)
        if self.display.mesh.name is not None:
            self.setWindowTitle(self.display.mesh.name)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        speed_d = 0.1
        if delta > 0:
            self.display.camera.distance -= speed_d
        else:
            self.display.camera.distance += speed_d

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.mouse_down_pos = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() == QtCore.Qt.MouseButton.LeftButton:
            dx = event.pos().x() - self.mouse_down_pos.x()
            dy = event.pos().y() - self.mouse_down_pos.y()
            speed_x = 0.01
            speed_y = 0.01
            self.display.camera.mouse_y(dy * speed_y)
            self.display.camera.mouse_x(dx * speed_x)
            self.mouse_down_pos = event.pos()


_app = None


def show_mesh(mesh: Mesh, dt=1 / 60):
    global _app
    camera = SphericalCamera()
    if _app is None:
        _app = QtWidgets.QApplication(sys.argv)
    viewer = Viewer(mesh, dt, camera)
    viewer.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    viewer.show()
    loop = QtCore.QEventLoop()

    def on_destroyed():
        loop.quit()

    viewer.destroyed.connect(on_destroyed)
    loop.exec()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_path", type=str, help="Path to the mesh file (OBJ format)")
    args = parser.parse_args()
    mesh_path = Path(args.mesh_path)
    mesh_trimesh: trimesh.Trimesh = trimesh.load(mesh_path)
    vertices = torch.tensor(mesh_trimesh.vertices, dtype=trigrad.precision, device="cuda")
    indices = torch.tensor(mesh_trimesh.faces, dtype=torch.int32, device="cuda")
    colors = torch.rand_like(vertices)
    opacities = torch.ones_like(vertices[:, 0])
    mesh = Mesh(vertices=vertices, indices=indices, colors=colors, opacities=opacities, name=mesh_path.name)
    show_mesh(mesh)
