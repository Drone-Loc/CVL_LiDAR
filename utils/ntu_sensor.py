"""Camera/LiDAR calibration helpers shared by NTU training and MPH inference.

The calibration files produced by direct_visual_lidar_calibration store
``T_lidar_camera`` as::

    p_lidar = R_lidar_camera @ p_camera + t_lidar_camera

where the quaternion is ordered ``qx, qy, qz, qw``.  The model projects in the
opposite direction, so this module performs the inversion in one place.
"""

from dataclasses import dataclass
import ast
import math
from pathlib import Path

import numpy as np


def _config_tuple(config, section, option, length, fallback=None):
    if config.has_option(section, option):
        value = ast.literal_eval(config.get(section, option))
    elif fallback is not None:
        value = fallback
    else:
        raise KeyError(f"Missing [{section}] {option}")
    value = tuple(float(v) for v in value)
    if len(value) != length:
        raise ValueError(f"[{section}] {option} must contain {length} values, got {len(value)}")
    return value


def quaternion_xyzw_to_rotation(quaternion):
    """Return a 3x3 rotation matrix for an ``(qx, qy, qz, qw)`` quaternion."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("Calibration quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def yaw_rotation(yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass(frozen=True)
class SensorCalibration:
    camera_model: str
    intrinsics: tuple
    distortion_coeffs: tuple
    T_lidar_camera: tuple
    align_lidar_to_camera_yaw: bool = True

    @classmethod
    def from_config(cls, config, section):
        camera_model = config.get(section, "camera_model", fallback="plumb_bob").strip().lower()
        if camera_model != "plumb_bob":
            raise ValueError(f"Unsupported [{section}] camera_model={camera_model!r}; expected 'plumb_bob'")
        return cls(
            camera_model=camera_model,
            intrinsics=_config_tuple(config, section, "intrinsics", 4),
            distortion_coeffs=_config_tuple(
                config, section, "distortion_coeffs", 5, fallback=(0.0, 0.0, 0.0, 0.0, 0.0)
            ),
            T_lidar_camera=_config_tuple(config, section, "T_lidar_camera", 7),
            align_lidar_to_camera_yaw=config.getboolean(
                section, "align_lidar_to_camera_yaw", fallback=True
            ),
        )

    @property
    def R_lidar_camera(self):
        return quaternion_xyzw_to_rotation(self.T_lidar_camera[3:])

    @property
    def t_lidar_camera(self):
        return np.asarray(self.T_lidar_camera[:3], dtype=np.float64)

    @property
    def R_camera_lidar(self):
        return self.R_lidar_camera.T

    @property
    def t_camera_lidar(self):
        return -self.R_camera_lidar @ self.t_lidar_camera

    @property
    def camera_forward_yaw_lidar_rad(self):
        """Yaw of the camera optical +z axis expressed in the raw LiDAR frame."""
        camera_forward_lidar = self.R_lidar_camera[:, 2]
        return math.atan2(float(camera_forward_lidar[1]), float(camera_forward_lidar[0]))

    @property
    def lidar_to_model_yaw_rad(self):
        """Rotate raw LiDAR XY so model +x follows the camera's horizontal heading."""
        if not self.align_lidar_to_camera_yaw:
            return 0.0
        return -self.camera_forward_yaw_lidar_rad

    @property
    def model_heading_offset_rad(self):
        """Offset added to raw LiDAR/SLAM yaw to obtain the model-frame heading."""
        if not self.align_lidar_to_camera_yaw:
            return 0.0
        return self.camera_forward_yaw_lidar_rad

    def projection_extrinsic(self):
        """Return ``R,t`` used after the model's nominal camera-axis conversion.

        Model points are first yaw-normalized and then internally converted from
        ``(forward, left, up)`` to optical ``(right, down, forward)`` axes.  This
        composes both operations with the full calibrated transform so that the
        projection remains exactly equivalent to ``T_camera_lidar``.
        """
        # p_model = R_model_lidar p_lidar, so p_lidar = R_lidar_model p_model.
        R_model_lidar = yaw_rotation(self.lidar_to_model_yaw_rad)
        R_lidar_model = R_model_lidar.T
        # Nominal LiDAR/model -> optical-camera axis conversion.
        A = np.array([[0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [1.0, 0.0, 0.0]], dtype=np.float64)
        R_after_nominal = self.R_camera_lidar @ R_lidar_model @ A.T
        return R_after_nominal.astype(np.float32), self.t_camera_lidar.astype(np.float32)

    def scaled_camera_matrix(self, source_size, destination_size):
        """Build K after a resize-only transform.

        ``source_size`` is ``(width, height)`` and ``destination_size`` is
        ``(height, width)`` to match the image tensor configuration.
        """
        src_w, src_h = source_size
        dst_h, dst_w = destination_size
        fx, fy, cx, cy = self.intrinsics
        sx, sy = float(dst_w) / float(src_w), float(dst_h) / float(src_h)
        return np.array([[fx * sx, 0.0, cx * sx],
                         [0.0, fy * sy, cy * sy],
                         [0.0, 0.0, 1.0]], dtype=np.float32)


def rotate_points_to_model(points, yaw_rad):
    """Yaw-rotate ``[..., x, y, z, ...]`` points without changing their type."""
    if abs(float(yaw_rad)) < 1e-12:
        return points
    rotated = points.clone() if hasattr(points, "clone") else points.copy()
    x = points[..., 0].clone() if hasattr(points[..., 0], "clone") else points[..., 0].copy()
    y = points[..., 1].clone() if hasattr(points[..., 1], "clone") else points[..., 1].copy()
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    rotated[..., 0] = c * x - s * y
    rotated[..., 1] = s * x + c * y
    return rotated


def resolve_dataset_path(configured_path, repository_dir, relative_fallback):
    """Use the configured Docker path, falling back to the local sibling dataset."""
    configured = Path(configured_path)
    if configured.exists():
        return configured
    fallback = Path(repository_dir).resolve().parent / relative_fallback
    return fallback if fallback.exists() else configured
