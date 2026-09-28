#!/usr/bin/env python3
import argparse
import json
import math
import pickle
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from utils.ntu_sensor import SensorCalibration

# ---------------------------------------------------------------------------
# User-editable defaults (map frame and pose I/O match mph_ros_test.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = SCRIPT_DIR / "calib"

MAP_DIR = "./maps/mph"
MAP_NAME = "mph_manual_viz.png"
TILE_GLOB = "*_manual_tile.pkl"
MAP_PPM = 20.0
ORIGIN_PX = (512.0, 776.0)
INITIAL_POSE = (8.3, 8.3, 0.0)    # x [m], y [m], yaw [deg]
POSE_FRAME_ID = "map"

PITCH_DOWN_DEG = 20.5                 # sensor mounting tilt compensation
LIDAR_MAX_RANGE_M = 15.0
LIDAR_Z_MIN_M = 0.3                   # after yaw+pitch rotation, sensor frame
LIDAR_Z_MAX_M = 3.0
BEV_SPAN_M = 32.0                     # BEV covers [-16, 16] m
BEV_RES = 574                         # BEV pixels (574 px / 32 m)


# ---------------------------------------------------------------------------
# Outline map
# ---------------------------------------------------------------------------
class OutlineMap:
    """Wall raster + distance transform + cached derived fields."""

    def __init__(self, wall, ppm, origin_px):
        self.wall = wall                       # (H, W) bool
        self.ppm = float(ppm)
        self.origin_px = (float(origin_px[0]), float(origin_px[1]))
        self.dt = ndimage.distance_transform_edt(~wall) / self.ppm   # metres
        self.grad_row, self.grad_col = np.gradient(self.dt)
        self._cache = {}

    @classmethod
    def load(cls, path, ppm, origin_px=None):
        wall = np.asarray(Image.open(path).convert("L")) > 127
        if origin_px is None:
            origin_px = detect_bottom_right_corner(wall)
        return cls(wall, ppm, origin_px)

    @property
    def shape(self):
        return self.wall.shape

    def likelihood_field(self, sigma_m, pad):
        key = (round(sigma_m, 4), pad)
        if key not in self._cache:
            lf = np.exp(-0.5 * (self.dt / sigma_m) ** 2).astype(np.float32)
            self._cache[key] = np.pad(lf, pad, mode="constant", constant_values=0.0)
        return self._cache[key]

    def wall_dilated(self, pad, thickness_m=0.3):
        key = ("wall", pad)
        if key not in self._cache:
            r = max(1, int(round(0.5 * (thickness_m * self.ppm - 1))))
            dil = ndimage.binary_dilation(self.wall, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
            self._cache[key] = np.pad(dil, pad, mode="constant", constant_values=False)
        return self._cache[key]

    def world_to_pixel(self, xw, yw):
        ox, oy = self.origin_px
        return ox - yw * self.ppm, oy - xw * self.ppm       # col, row


def detect_bottom_right_corner(wall):
    """Outline bottom-right corner: densest wall column (right half) x row (bottom half)."""
    ys, xs = np.nonzero(wall)
    c0, c1 = xs.min(), xs.max()
    r0, r1 = ys.min(), ys.max()
    right_col = (c0 + c1) // 2 + int(np.argmax(wall[:, (c0 + c1) // 2:].sum(axis=0)))
    bottom_row = (r0 + r1) // 2 + int(np.argmax(wall[(r0 + r1) // 2:, :].sum(axis=1)))
    return right_col, bottom_row


# ---------------------------------------------------------------------------
# Cloud -> BEV scan points (body frame, metres)
# ---------------------------------------------------------------------------
def cloud_to_bev_points(xyz, yaw_deg, pitch_down_deg=PITCH_DOWN_DEG,
                        max_range=LIDAR_MAX_RANGE_M, z_min=LIDAR_Z_MIN_M, z_max=LIDAR_Z_MAX_M,
                        span=BEV_SPAN_M, res=BEV_RES):
    """Raw (N,3+) LiDAR points -> deduplicated 2-D BEV cell centres [x fwd, y left]."""
    xyz = xyz[np.isfinite(xyz[:, :3]).all(axis=1)]
    xyz = xyz[np.hypot(xyz[:, 0], xyz[:, 1]) <= max_range]
    x, y, z = xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64), xyz[:, 2].astype(np.float64)
    # yaw to model frame (reversed LiDAR mount)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    x, y = c * x - s * y, s * x + c * y
    # pitch the cloud nose-down (compensates the upward-tilted sensor)
    ct, st = math.cos(math.radians(pitch_down_deg)), math.sin(math.radians(pitch_down_deg))
    x, z = ct * x + st * z, -st * x + ct * z
    m = (z >= z_min) & (z <= z_max) & (np.abs(x) < span / 2) & (np.abs(y) < span / 2)
    if not np.any(m):
        return np.zeros((0, 2))
    v = span / res
    rows = np.clip((res - 1 - np.floor((x[m] + span / 2) / v)).astype(int), 0, res - 1)
    cols = np.clip(np.floor((-y[m] + span / 2) / v).astype(int), 0, res - 1)
    occ = np.zeros((res, res), bool)
    occ[rows, cols] = True
    rr, cc = np.nonzero(occ)
    fwd = (res - 0.5 - rr) * v - span / 2
    lat = (cc + 0.5) * v - span / 2
    return np.stack([fwd, -lat], axis=1)


def transform_points(pts, pose):
    x0, y0, yaw_deg = pose
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    xw = x0 + c * pts[:, 0] - s * pts[:, 1]
    yw = y0 + s * pts[:, 0] + c * pts[:, 1]
    return np.stack([xw, yw], axis=1)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Matching: coarse-to-fine grid search with free-space term
# ---------------------------------------------------------------------------
def build_ray_samples(pts, bin_deg=1.5, step_m=0.25, margin_m=0.4, skip_m=0.3):
    """Sample points along the ray to the farthest return per angle bin (M, S, 2)."""
    rng = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    bins = np.floor((ang + 180.0) / bin_deg).astype(int)
    order = np.lexsort((-rng, bins))
    first = np.r_[True, bins[order][1:] != bins[order][:-1]]
    far = pts[order][first]
    far_rng = rng[order][first]
    keep = far_rng > skip_m + margin_m
    far, far_rng = far[keep], far_rng[keep]
    n_s = int(math.ceil((far_rng.max() - skip_m - margin_m) / step_m)) + 1
    ts = np.linspace(0.0, 1.0, n_s)
    f_lo = skip_m / far_rng
    f_hi = (far_rng - margin_m) / far_rng
    frac = f_lo[:, None] + (f_hi - f_lo)[:, None] * ts[None, :]
    return far[:, None, :] * frac[:, :, None]


def grid_search(omap, pts, rays, center, win_xy_m, win_yaw_deg, step_xy_m, step_yaw_deg, sigma_m):
    """score(dx,dy,dyaw) = mean LF(points) * (1 - fraction of rays crossing a wall)."""
    pad = int(math.ceil((win_xy_m + BEV_SPAN_M) * omap.ppm)) + 2
    lf = omap.likelihood_field(sigma_m, pad)
    wall = omap.wall_dilated(pad) if rays is not None else None
    nx = int(round(win_xy_m / step_xy_m))
    ny = int(round(win_yaw_deg / step_yaw_deg))
    dxs = np.arange(-nx, nx + 1) * step_xy_m
    dys = dxs.copy()
    dyaws = np.arange(-ny, ny + 1) * step_yaw_deg
    x0, y0, yaw0 = center
    drow = np.rint(-dxs * omap.ppm).astype(np.int64)
    dcol = np.rint(-dys * omap.ppm).astype(np.int64)
    scores = np.empty((len(dyaws), len(dxs), len(dys)))
    n = len(pts)
    if rays is not None:
        M, S, _ = rays.shape
        rays_flat = rays.reshape(-1, 2)
    for k, dyaw in enumerate(dyaws):
        pose_k = (x0, y0, yaw0 + dyaw)
        pw = transform_points(pts, pose_k)
        col, row = omap.world_to_pixel(pw[:, 0], pw[:, 1])
        rows = np.rint(row).astype(np.int64)[:, None] + pad + drow[None, :]
        cols = np.rint(col).astype(np.int64)[:, None] + pad + dcol[None, :]
        if rays is not None:
            rw = transform_points(rays_flat, pose_k)
            rcol, rrow = omap.world_to_pixel(rw[:, 0], rw[:, 1])
            rrows = np.rint(rrow).astype(np.int64)[:, None] + pad + drow[None, :]
            rcols = np.rint(rcol).astype(np.int64)[:, None] + pad + dcol[None, :]
        acc = np.empty((len(dxs), len(dys)))
        for ix in range(len(dxs)):
            acc[ix] = lf[rows[:, ix][:, None], cols].sum(axis=0) / n
            if rays is not None:
                hit = wall[rrows[:, ix][:, None], rcols].reshape(M, S, -1).any(axis=1)
                acc[ix] *= 1.0 - hit.sum(axis=0) / M
        scores[k] = acc
    return scores, dxs, dys, dyaws


def local_maxima(scores, dxs, dys, dyaws, top_k, min_sep_m, min_sep_deg):
    filt = ndimage.maximum_filter(scores, size=3, mode="nearest")
    cand = np.argwhere((scores == filt) & (scores > 0.2 * scores.max()))
    cand = sorted(cand, key=lambda idx: -scores[tuple(idx)])
    picked = []
    for k, ix, iy in cand:
        p = (dxs[ix], dys[iy], dyaws[k], scores[k, ix, iy])
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_sep_m
               or abs(p[2] - q[2]) >= min_sep_deg for q in picked):
            picked.append(p)
        if len(picked) >= top_k:
            break
    return picked


def _sample_dt(omap, col, row):
    coords = np.stack([row, col])
    d = ndimage.map_coordinates(omap.dt, coords, order=1, mode="nearest")
    gr = ndimage.map_coordinates(omap.grad_row, coords, order=1, mode="nearest")
    gc = ndimage.map_coordinates(omap.grad_col, coords, order=1, mode="nearest")
    return d, gr, gc


def _residuals_jacobian(omap, pts, pose):
    x0, y0, yaw_deg = pose
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)
    xb, yb = pts[:, 0], pts[:, 1]
    xw = x0 + c * xb - s * yb
    yw = y0 + s * xb + c * yb
    col, row = omap.world_to_pixel(xw, yw)
    d, gr, gc = _sample_dt(omap, col, row)
    ddx = -omap.ppm * gr
    ddy = -omap.ppm * gc
    dxw_dth = -s * xb - c * yb
    dyw_dth = c * xb - s * yb
    J = np.stack([ddx, ddy, ddx * dxw_dth + ddy * dyw_dth], axis=1)
    return d, J


def refine_lm(omap, pts, pose, iters=40, trim_m=(1.0, 0.6, 0.4), huber_m=0.15, inlier_m=0.3):
    """Trimmed-Huber Levenberg-Marquardt on the distance-transform residuals."""
    x = np.array([pose[0], pose[1], math.radians(pose[2])])
    lam = 1e-3

    def cost_and_system(xv, trim):
        d, J = _residuals_jacobian(omap, pts, (xv[0], xv[1], math.degrees(xv[2])))
        w = np.where(d <= trim, 1.0, 0.0)
        big = d > huber_m
        w[big] *= huber_m / np.maximum(d[big], 1e-9)
        return float(np.sum(w * d * d)), d, J, w

    stage = 0
    trim = trim_m[stage]
    cost, d, J, w = cost_and_system(x, trim)
    for _ in range(iters):
        H = J.T @ (J * w[:, None])
        g = J.T @ (w * d)
        step_ok = False
        for _ in range(8):
            A = H + lam * np.diag(np.diag(H) + 1e-9)
            try:
                delta = -np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            x_new = x + delta
            cost_new, d_new, J_new, w_new = cost_and_system(x_new, trim)
            if cost_new < cost:
                x, cost, d, J, w = x_new, cost_new, d_new, J_new, w_new
                lam = max(lam / 3.0, 1e-6)
                step_ok = True
                break
            lam *= 10.0
        converged = (not step_ok) or (np.abs(delta[:2]).max() < 1e-4 and abs(delta[2]) < 1e-5)
        if converged:
            if stage + 1 < len(trim_m):
                stage += 1
                trim = trim_m[stage]
                cost, d, J, w = cost_and_system(x, trim)
                lam = 1e-3
                continue
            break

    H = J.T @ (J * w[:, None])
    n_eff = int(np.sum(w > 0))
    sigma2 = cost / max(n_eff - 3, 1)
    try:
        cov = sigma2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    inlier_ratio = float(np.mean(d <= inlier_m))
    pose_out = (float(x[0]), float(x[1]), float(wrap_deg(math.degrees(x[2]))))
    return pose_out, cov, inlier_ratio


def score_pose(omap, pts, pose, sigma_m=0.25):
    pw = transform_points(pts, pose)
    col, row = omap.world_to_pixel(pw[:, 0], pw[:, 1])
    d = ndimage.map_coordinates(omap.dt, np.stack([row, col]), order=1, mode="nearest")
    return float(np.mean(np.exp(-0.5 * (d / sigma_m) ** 2)))


def visibility_violation(omap, pts, pose, step_m=0.1, margin_m=0.4, skip_m=0.3):
    """Fraction of scan points whose sensor->point ray crosses a map wall."""
    pad = int(math.ceil(BEV_SPAN_M * omap.ppm)) + 2
    wall = omap.wall_dilated(pad)
    pw = transform_points(pts, pose)
    col, row = omap.world_to_pixel(pw[:, 0], pw[:, 1])
    c0, r0 = omap.world_to_pixel(np.array([pose[0]]), np.array([pose[1]]))
    rng = np.hypot(pts[:, 0], pts[:, 1])
    f_lo = np.clip(skip_m / np.maximum(rng, 1e-6), 0.0, 1.0)
    f_hi = np.clip((rng - margin_m) / np.maximum(rng, 1e-6), 0.0, 1.0)
    n_s = int(math.ceil(rng.max() / step_m)) + 1
    ts = np.linspace(0.0, 1.0, n_s)
    frac = f_lo[:, None] + (f_hi - f_lo)[:, None] * ts[None, :]
    rr = np.rint(r0[0] + (row - r0[0])[:, None] * frac).astype(np.int64) + pad
    cc = np.rint(c0[0] + (col - c0[0])[:, None] * frac).astype(np.int64) + pad
    rr = np.clip(rr, 0, wall.shape[0] - 1)
    cc = np.clip(cc, 0, wall.shape[1] - 1)
    hit = wall[rr, cc].any(axis=1)
    hit &= f_hi > f_lo
    return float(np.mean(hit))


def localize(omap, pts, init_pose, win_xy_m=6.0, win_yaw_deg=12.0, top_k=3):
    """Full solve: L1 grid 0.5 m / 2 deg -> L2 grid 0.1 m / 0.5 deg -> LM."""
    t0 = time.perf_counter()
    rays = build_ray_samples(pts)
    s1, dxs, dys, dyaws = grid_search(omap, pts, rays, init_pose, win_xy_m, win_yaw_deg, 0.5, 2.0, 0.6)
    modes = local_maxima(s1, dxs, dys, dyaws, top_k, min_sep_m=0.6, min_sep_deg=2.4)

    results = []
    for dx, dy, dyaw, _ in modes:
        c1 = (init_pose[0] + dx, init_pose[1] + dy, init_pose[2] + dyaw)
        s2, dxs2, dys2, dyaws2 = grid_search(omap, pts, rays, c1, 0.8, 3.0, 0.1, 0.5, 0.3)
        k, ix, iy = np.unravel_index(np.argmax(s2), s2.shape)
        c2 = (c1[0] + dxs2[ix], c1[1] + dys2[iy], c1[2] + dyaws2[k])
        pose, cov, inl = refine_lm(omap, pts, c2)
        lf = score_pose(omap, pts, pose)
        viol = visibility_violation(omap, pts, pose)
        results.append(dict(pose=pose, cov=cov, inlier_ratio=inl,
                            lf_score=lf, violation=viol, score=lf * (1.0 - viol)))
    results.sort(key=lambda r: -r["score"])
    best = results[0]
    others = [r["score"] for r in results[1:]
              if math.hypot(r["pose"][0] - best["pose"][0], r["pose"][1] - best["pose"][1]) > 0.3
              or abs(wrap_deg(r["pose"][2] - best["pose"][2])) > 1.0]
    best["ambiguity"] = (max(others) if others else 0.0) / max(best["score"], 1e-9)
    best["time_s"] = time.perf_counter() - t0
    return best


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def draw_pose_arrow(draw, omap, pose, color, length_m=3.0):
    col, row = omap.world_to_pixel(np.array([pose[0]]), np.array([pose[1]]))
    x, y = col[0], row[0]
    th = math.radians(pose[2])
    L = length_m * omap.ppm
    d = np.array([-math.sin(th), -math.cos(th)])
    n = np.array([math.cos(th), -math.sin(th)])
    tail = np.array([x, y])
    tip = tail + d * L
    neck = tip - d * 0.3 * L
    w2, h2 = max(1.0, 0.05 * L) / 2 * 2, 0.11 * L
    poly = [tail + n * w2 / 2, neck + n * w2 / 2, neck + n * h2, tip,
            neck - n * h2, neck - n * w2 / 2, tail - n * w2 / 2]
    draw.polygon([tuple(v) for v in poly], fill=color)
    r = max(3.0, 0.08 * L)
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color)


def save_result_images(omap, pts, init_pose, est_pose, output_path):
    result_path = Path(output_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # 1) full map with init (red) / estimate (green) arrows
    base = Image.fromarray((omap.wall * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(base)
    draw_pose_arrow(draw, omap, init_pose, (255, 60, 60))
    draw_pose_arrow(draw, omap, est_pose, (60, 255, 60))
    base.save(result_path)
    # 2) body-frame overlay: scan as-recorded (forward = up), map rotated onto it
    x0, y0, yaw = est_pose
    half = int(round(22.0 * omap.ppm))
    big = int(math.ceil(half * math.sqrt(2.0))) + 2
    pad = big + 2
    wall_pad = np.pad(omap.wall, pad, mode="constant", constant_values=False)
    cx, cy = omap.world_to_pixel(np.array([x0]), np.array([y0]))
    c0 = int(round(float(cx[0]))) - big + pad
    r0 = int(round(float(cy[0]))) - big + pad
    crop = wall_pad[r0:r0 + 2 * big, c0:c0 + 2 * big]
    img = Image.fromarray((crop * 255).astype(np.uint8))
    img = img.rotate(-yaw, resample=Image.NEAREST, expand=False)
    img = img.crop((big - half, big - half, big + half, big + half)).convert("RGB")
    px = np.asarray(img).copy()
    col = np.rint(half - pts[:, 1] * omap.ppm).astype(int)
    row = np.rint(half - pts[:, 0] * omap.ppm).astype(int)
    ok = (col >= 0) & (col < px.shape[1]) & (row >= 0) & (row < px.shape[0])
    px[row[ok], col[ok]] = (60, 255, 60)
    img = Image.fromarray(px)
    draw = ImageDraw.Draw(img)
    c = half
    L = 3.0 * omap.ppm
    draw.polygon([(c - 2, c), (c - 2, c - 0.7 * L), (c - 0.11 * L, c - 0.7 * L), (c, c - L),
                  (c + 0.11 * L, c - 0.7 * L), (c + 2, c - 0.7 * L), (c + 2, c)], fill=(60, 255, 60))
    draw.ellipse([c - 4, c - 4, c + 4, c + 4], fill=(60, 255, 60))
    overlay_path = result_path.with_name(
        f"{result_path.stem}_overlay{result_path.suffix}")
    img.save(overlay_path)
    return result_path, overlay_path


# ---------------------------------------------------------------------------
# ROS message conversion
# ---------------------------------------------------------------------------
def livox_custommsg_to_xyz(msg):
    if not msg.points:
        raise ValueError("Empty Livox CustomMsg")
    return np.asarray([(p.x, p.y, p.z) for p in msg.points], dtype=np.float32)


def messages_to_xyz(msgs, to_xyz):
    """Merge an accumulated LiDAR window into a single (N, 3) cloud."""
    batches = []
    for msg in msgs:
        try:
            batches.append(to_xyz(msg))
        except ValueError:
            continue
    if not batches:
        raise ValueError("Accumulated LiDAR window contains no points")
    return np.concatenate(batches, axis=0)


def pointcloud2_to_xyz(msg):
    offsets = {f.name: f.offset for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in offsets:
            raise ValueError(f"PointCloud2 missing field '{name}'")
    dtype = np.dtype({"names": ["x", "y", "z"],
                      "formats": ["<f4", "<f4", "<f4"],
                      "offsets": [offsets["x"], offsets["y"], offsets["z"]],
                      "itemsize": msg.point_step})
    n = msg.width * msg.height
    rec = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------
def resolve_map_path(map_dir):
    """Accept either the map directory used by mph_ros_test.py or a raster path."""
    path = Path(map_dir)
    if path.is_dir():
        path = path / MAP_NAME
    if not path.exists():
        raise FileNotFoundError(f"Outline map does not exist: {path}")
    return path


def resolve_map_ppm(map_path):
    """Read the raster resolution from the tile that ships with the map."""
    for tile_path in sorted(map_path.parent.glob(TILE_GLOB)):
        with tile_path.open("rb") as stream:
            tile = pickle.load(stream)
        ppm = tile.get("ppm")
        if ppm:
            return float(ppm), tile_path
    return MAP_PPM, None


def drone_lidar_to_model_yaw_deg(drone_id):
    """Yaw that aligns raw LiDAR XY with the model frame, from calib/calib_<id>.json."""
    calibration_path = CALIBRATION_DIR / f"calib_{int(drone_id)}.json"
    with calibration_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    camera = data["camera"]
    calibration = SensorCalibration(
        camera_model=camera["camera_model"],
        intrinsics=tuple(camera["intrinsics"]),
        distortion_coeffs=tuple(camera["distortion_coeffs"]),
        T_lidar_camera=tuple(data["results"]["T_lidar_camera"]),
        align_lidar_to_camera_yaw=True,
    )
    return math.degrees(calibration.lidar_to_model_yaw_rad), calibration_path


def pose3dof_to_pose_stamped(x, y, yaw, z=0.0):
    import rospy
    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = POSE_FRAME_ID
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.orientation.w = math.cos(yaw / 2.0)
    return msg


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drone_id", type=int, choices=range(1, 7), default=2,
                        help="Drone number used to load calib/calib_<id>.json")
    parser.add_argument("--lidar_topic", default="/livox/lidar",
                        help="LiDAR topic (livox_ros_driver2/CustomMsg or sensor_msgs/PointCloud2)")
    parser.add_argument("--msg_type", choices=["custom", "pointcloud2"], default="custom",
                        help="LiDAR message type on --lidar_topic")
    parser.add_argument("--pose_topic", default=None,
                        help="PoseStamped output topic; defaults to /agent00X/oneshot_localization_result")
    parser.add_argument("--initial_pose", nargs=3, type=float, default=list(INITIAL_POSE),
                        metavar=("X", "Y", "YAW_DEG"),
                        help=("Initial map pose: x forward and y left in meters, "
                              "yaw CCW-positive in degrees; e.g. 8.3 8.3 45"))
    parser.add_argument("--map_dir", default=MAP_DIR,
                        help=f"Map directory holding {MAP_NAME}, or a raster path")
    parser.add_argument("--ppm", type=float, default=None,
                        help=f"Map resolution in px/m; read from {TILE_GLOB} when omitted")
    parser.add_argument("--origin_px", nargs=2, type=float, default=list(ORIGIN_PX),
                        metavar=("COL", "ROW"),
                        help="Pixel of world (0, 0): the outline's bottom-right corner")
    parser.add_argument("--window", nargs=2, type=float, default=(6.0, 30.0),
                        metavar=("XY_M", "YAW_DEG"), help="search half-window around the init pose")
    parser.add_argument("--track_window", nargs=2, type=float, default=(3.0, 10.0),
                        metavar=("XY_M", "YAW_DEG"), help="window in --continuous mode after the first fix")
    parser.add_argument("--pitch_down_deg", type=float, default=PITCH_DOWN_DEG)
    parser.add_argument("--queue_size", type=int, default=10)
    parser.add_argument("--lidar_accumulation_sec", type=float, default=0.3,
                        help="LiDAR accumulation window in seconds")
    parser.add_argument("--continuous", action="store_true",
                        help="keep localizing every window (previous estimate becomes the init)")
    parser.add_argument("--output_path", default="results/oneshot_localization.png",
                        help="Result image; the body-frame overlay reuses this stem")
    parser.add_argument("--exit_after_publish", action="store_true",
                        help="one-shot mode: exit after publishing instead of latching")
    args = parser.parse_args()
    if args.lidar_accumulation_sec <= 0.0:
        parser.error("--lidar_accumulation_sec must be positive")
    if args.pose_topic is None:
        args.pose_topic = f"/agent{args.drone_id:03d}/oneshot_localization_result"
    return args


class MPHNonLearnROSLocalizer:
    def __init__(self, args):
        import rospy
        from geometry_msgs.msg import PoseStamped

        self.rospy = rospy
        self.args = args
        self.lock = threading.Lock()
        self.processing = False
        self.finished = False
        self.frame_index = 0
        self.lidar_buffer = []

        self.map_path = resolve_map_path(args.map_dir)
        if args.ppm is None:
            args.ppm, tile_path = resolve_map_ppm(self.map_path)
            rospy.loginfo("Map resolution %g px/m from %s",
                          args.ppm, tile_path or "built-in default")
        self.omap = OutlineMap.load(str(self.map_path), args.ppm, args.origin_px)
        self.current_pose = tuple(float(v) for v in args.initial_pose)
        rospy.loginfo("Map: %s (%dx%d px @ %g ppm, origin pixel col=%.1f row=%.1f)",
                      self.map_path, self.omap.shape[1], self.omap.shape[0], args.ppm,
                      self.omap.origin_px[0], self.omap.origin_px[1])

        self.pose_publisher = rospy.Publisher(
            args.pose_topic, PoseStamped, queue_size=1, latch=True
        )
        if args.msg_type == "custom":
            from livox_ros_driver2.msg import CustomMsg
            self.subscriber = rospy.Subscriber(args.lidar_topic, CustomMsg,
                                               self.on_cloud, queue_size=args.queue_size)
            self.to_xyz = livox_custommsg_to_xyz
        else:
            from sensor_msgs.msg import PointCloud2
            self.subscriber = rospy.Subscriber(args.lidar_topic, PointCloud2,
                                               self.on_cloud, queue_size=args.queue_size)
            self.to_xyz = pointcloud2_to_xyz

    # -- LiDAR accumulation (same windowing as mph_ros_test.py) --------------
    def _append_lidar_locked(self, lidar_msg):
        if any(buffered_msg is lidar_msg for _, buffered_msg in self.lidar_buffer):
            return

        lidar_time = lidar_msg.header.stamp.to_sec()
        if self.lidar_buffer and lidar_time <= self.lidar_buffer[-1][0]:
            self.rospy.logwarn(
                "LiDAR timestamp did not increase; restarting accumulation window"
            )
            self.lidar_buffer.clear()
        self.lidar_buffer.append((lidar_time, lidar_msg))
        self._prune_lidar_buffer_locked(lidar_time)

    def _prune_lidar_buffer_locked(self, latest_time):
        """Keep a bounded window that still covers the requested duration."""
        target_start_time = latest_time - self.args.lidar_accumulation_sec
        keep_from = 0
        for index, (stamp, _) in enumerate(self.lidar_buffer):
            if stamp > target_start_time:
                keep_from = max(0, index - 1)
                break
        if keep_from:
            del self.lidar_buffer[:keep_from]

    def _take_accumulated_lidar_locked(self, lidar_msg):
        self._append_lidar_locked(lidar_msg)
        lidar_end_time = lidar_msg.header.stamp.to_sec()
        eligible = [item for item in self.lidar_buffer if item[0] <= lidar_end_time]
        if not eligible:
            return None

        target_start_time = lidar_end_time - self.args.lidar_accumulation_sec
        if eligible[0][0] > target_start_time:
            return None

        # Include the closest message at or before the target start so the
        # accumulated data covers at least the requested duration.
        start_index = 0
        for index, (stamp, _) in enumerate(eligible):
            if stamp > target_start_time:
                break
            start_index = index
        window = eligible[start_index:]
        self.lidar_buffer = [
            item for item in self.lidar_buffer if item[0] > lidar_end_time
        ]
        return window

    def on_cloud(self, msg):
        with self.lock:
            if self.processing or self.finished:
                return
            lidar_window = self._take_accumulated_lidar_locked(msg)
            if lidar_window is None:
                return
            self.processing = True
        try:
            lidar_msgs = [item[1] for item in lidar_window]
            xyz = messages_to_xyz(lidar_msgs, self.to_xyz)
            pts = cloud_to_bev_points(
                xyz, yaw_deg=self.args.yaw_to_model_deg,
                pitch_down_deg=self.args.pitch_down_deg)
            if len(pts) < 200:
                raise ValueError(f"Too few BEV points after filtering: {len(pts)}")

            first = self.frame_index == 0
            win = self.args.window if (first or not self.args.continuous) else self.args.track_window
            init = self.current_pose
            res = localize(self.omap, pts, init, win_xy_m=win[0], win_yaw_deg=win[1])
            x, y, yaw = res["pose"]
            yaw_rad = math.radians(yaw)

            published_pose = pose3dof_to_pose_stamped(x, y, yaw_rad)
            self.pose_publisher.publish(published_pose)

            self.rospy.loginfo(
                "frame %d: pose x=%.3f m y=%.3f m yaw=%.2f deg CCW (%.6f rad) | "
                "score=%.3f inlier=%.3f viol=%.3f amb=%.2f%s | N=%d t=%.2fs",
                self.frame_index, x, y, yaw, yaw_rad, res["score"], res["inlier_ratio"],
                res["violation"], res["ambiguity"],
                " AMBIGUOUS" if res["ambiguity"] > 0.9 else "",
                len(pts), res["time_s"])
            self.rospy.loginfo(
                "NWU pose: x=%.3f m, y=%.3f m, z=%.3f m, qz=%.6f, qw=%.6f",
                published_pose.pose.position.x,
                published_pose.pose.position.y,
                published_pose.pose.position.z,
                published_pose.pose.orientation.z,
                published_pose.pose.orientation.w)

            if first or not self.args.continuous:
                result_path, overlay_path = save_result_images(
                    self.omap, pts, init, res["pose"], self.args.output_path)
                self.rospy.loginfo("Saved: %s | %s", result_path, overlay_path)

            self.frame_index += 1
            if self.args.continuous:
                self.current_pose = res["pose"]
            else:
                with self.lock:
                    self.finished = True
                self.rospy.loginfo(
                    "Published latched geometry_msgs/PoseStamped: %s", self.args.pose_topic)
                if self.args.exit_after_publish:
                    self.rospy.signal_shutdown("One-shot localization completed")
        except Exception:
            self.rospy.logerr("Localization failed:\n%s", traceback.format_exc())
        finally:
            with self.lock:
                self.processing = False


def main():
    import rospy

    args = parse_args()
    args.yaw_to_model_deg, calibration_path = drone_lidar_to_model_yaw_deg(args.drone_id)

    rospy.init_node("mph_nonlearn_localizer", anonymous=False)
    rospy.loginfo("Drone ID: %d", args.drone_id)
    rospy.loginfo("Calibration: %s", calibration_path)
    localizer = MPHNonLearnROSLocalizer(args)
    rospy.loginfo("LiDAR topic: %s (%s)", args.lidar_topic,
                  "livox_ros_driver2/CustomMsg" if args.msg_type == "custom"
                  else "sensor_msgs/PointCloud2")
    rospy.loginfo("Initial map pose: x=%.3f m, y=%.3f m, yaw=%.3f deg CCW",
                  args.initial_pose[0], args.initial_pose[1], args.initial_pose[2])
    rospy.loginfo(
        "Pose topic: %s (geometry_msgs/PoseStamped, frame=%s, NWU, latched)",
        args.pose_topic, POSE_FRAME_ID)
    rospy.spin()
    return localizer


if __name__ == "__main__":
    main()
