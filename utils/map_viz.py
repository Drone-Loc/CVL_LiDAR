import numpy as np
import torch
import matplotlib.pyplot as plt
from osm.viz import Colormap

def map_viz(raster, *, to_uint8=True):
    """
    将 OrienterNet 的 raster/rasters 可视化为 RGB map。
    支持输入:
      - torch.Tensor / np.ndarray
      - shape: (3, H, W)  -> 语义栅格 (areas, ways, nodes)
      - shape: (H, W, 3)  -> 已经是 RGB
    返回:
      - RGB 图像: (H, W, 3), uint8(默认) 或 float32(0~1)
    """
    # 1) 转 numpy
    if isinstance(raster, torch.Tensor):
        raster = raster.detach().cpu().numpy()
    raster = np.asarray(raster)

    # 2) 如果已经是 RGB (H, W, 3)
    if raster.ndim == 3 and raster.shape[-1] == 3:
        rgb = raster
        # 若是 CHW 的 RGB (3,H,W)，下面会处理
        if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        # 统一到 float 0~1 or uint8
        if to_uint8:
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            if np.issubdtype(rgb.dtype, np.integer):
                rgb = rgb.astype(np.float32) / 255.0
            else:
                rgb = rgb.astype(np.float32)
        return rgb

    # 3) 语义栅格：期望 (3,H,W)
    if raster.ndim != 3 or raster.shape[0] < 2:
        raise ValueError(f"Expected rasters shape (3,H,W) or RGB (H,W,3), got {raster.shape}")

    # 确保是 int index（Colormap.apply 需要用作索引）
    if not np.issubdtype(raster.dtype, np.integer):
        raster = raster.astype(np.int64)

    # 4) 生成 RGB（float 0~1）
    rgb_f = Colormap.apply(raster)  # (H,W,3), float in 0~1 :contentReference[oaicite:2]{index=2}

    if to_uint8:
        return (np.clip(rgb_f, 0, 1) * 255).astype(np.uint8)
    else:
        return rgb_f.astype(np.float32)

def show_map(raster, save_path=None):
    img = map_viz(raster, to_uint8=False)  # 0~1 float
    plt.figure(figsize=(5, 5))
    plt.imshow(img, interpolation="none")
    plt.axis("off")

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()
    else:
        plt.close()
