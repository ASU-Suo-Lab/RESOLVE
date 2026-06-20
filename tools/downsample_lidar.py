#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from pathlib import Path
import random


# ============================================================
# IO
# ============================================================
def read_bin_kitti(bin_path: str, num_features: int = 4) -> np.ndarray:
    pts = np.fromfile(bin_path, dtype=np.float32)
    if pts.size % num_features != 0:
        raise ValueError(f"{bin_path}: size {pts.size} not divisible by {num_features}")
    return pts.reshape(-1, num_features)


def write_bin_kitti(bin_path: str, points: np.ndarray):
    out_dir = os.path.dirname(bin_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    points.astype(np.float32).tofile(bin_path)


# ============================================================
# Geometry: virtual rings from elevation
# ============================================================
def compute_elevation_deg(xyz: np.ndarray) -> np.ndarray:
    """
    z is UP.
    elevation = atan2(z, sqrt(x^2+y^2)) in degrees
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x * x + y * y) + 1e-6
    return np.degrees(np.arctan2(z, r))


def assign_virtual_rings_from_elev(
    elev_deg: np.ndarray,
    num_lines: int,
    fov_down: float,
    fov_up: float
) -> np.ndarray:
    elev = np.clip(elev_deg, fov_down, fov_up)
    bins = np.linspace(fov_down, fov_up, num_lines + 1)
    ring = np.digitize(elev, bins) - 1
    return np.clip(ring, 0, num_lines - 1).astype(np.int32)


def downsample_vertical(
    points: np.ndarray,
    num_lines: int,
    target_lines: int,
    fov_down: float,
    fov_up: float,
    start_offset: int = 0
) -> np.ndarray:
    """
    Keep every 'step' virtual ring to emulate fewer vertical beams.
    """
    if points is None or points.size == 0:
        return points

    assert num_lines % target_lines == 0, "num_lines must be divisible by target_lines"
    step = num_lines // target_lines

    elev = compute_elevation_deg(points[:, :3])
    ring = assign_virtual_rings_from_elev(elev, num_lines, fov_down, fov_up)
    keep = ((ring - start_offset) % step) == 0
    return points[keep]


def merge_two_clouds(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a is None or a.size == 0:
        return b if b is not None else np.zeros((0, 4), dtype=np.float32)
    if b is None or b.size == 0:
        return a
    return np.vstack([a, b])


# ============================================================
# Matplotlib check plots
# ============================================================
def save_ring_stats_png(points: np.ndarray,
                        num_lines: int,
                        fov_down: float,
                        fov_up: float,
                        out_png_prefix: str,
                        bins_hist: int = 200):
    """
    保存两张图：
      1) elevation histogram
      2) points per virtual ring
    """
    xyz = points[:, :3]
    elev = compute_elevation_deg(xyz)
    ring = assign_virtual_rings_from_elev(elev, num_lines, fov_down, fov_up)
    counts = np.bincount(ring, minlength=num_lines)

    # elevation hist
    plt.figure(figsize=(6, 4), dpi=160)
    plt.hist(elev, bins=bins_hist)
    plt.axvline(fov_down, linewidth=2, linestyle="--", label="fov_down")
    plt.axvline(fov_up, linewidth=2, linestyle="--", label="fov_up")
    plt.xlabel("Elevation angle (deg)")
    plt.ylabel("Count")
    plt.title("Elevation distribution (frame)")
    plt.legend()
    plt.tight_layout()
    p1 = f"{out_png_prefix}_elev_hist.png"
    plt.savefig(p1)
    plt.close()

    # points per ring
    plt.figure(figsize=(6, 4), dpi=160)
    plt.plot(np.arange(num_lines), counts, marker="o", linewidth=1)
    plt.xlabel("Virtual ring id")
    plt.ylabel("Point count")
    plt.title(f"Points per virtual ring (num_lines={num_lines})")
    plt.tight_layout()
    p2 = f"{out_png_prefix}_ring_counts.png"
    plt.savefig(p2)
    plt.close()

    return p1, p2


# ============================================================
# Open3D visualization
# ============================================================
def _to_o3d_pcd(points_xyz: np.ndarray, color=(0.6, 0.6, 0.6), max_points: int = 250000):
    if points_xyz.shape[0] > max_points:
        idx = np.random.choice(points_xyz.shape[0], max_points, replace=False)
        points_xyz = points_xyz[idx]

    pcd = o3d.geometry.PointCloud()
    xyz = points_xyz.astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    colors = np.tile(np.array(color, dtype=np.float64), (xyz.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_overlay_triplet(
    merged_128: np.ndarray,
    merged_64: np.ndarray,
    merged_16: np.ndarray,
    title: str,
    max_points: int
):
    """
    红：原始合并(128b+128)
    绿：64 down 合并
    蓝：16 down 合并
    """
    geoms = []
    geoms.append(_to_o3d_pcd(merged_128[:, :3], color=(1.0, 0.0, 0.0), max_points=max_points))
    geoms.append(_to_o3d_pcd(merged_64[:, :3],  color=(0.0, 1.0, 0.0), max_points=max_points))
    geoms.append(_to_o3d_pcd(merged_16[:, :3],  color=(0.0, 0.0, 1.0), max_points=max_points))

    o3d.visualization.draw(
        geoms,
    )


# ============================================================
# Core processing per subdir
# ============================================================
def process_one_subdir(
    sub_path: Path,
    num_lines: int,
    fov_down: float,
    fov_up: float,
    start_offset: int,
    targets: list, 
    do_viz: bool,
    viz_samples: int,
    viz_save: bool,
    viz_max_points: int
):
    dir_128b = sub_path / "ouster128b_label"
    dir_128  = sub_path / "ouster128_label"

    if not dir_128b.exists() or not dir_128.exists():
        print(f"⚠️ 跳过（缺少 ouster128b_label 或 ouster128_label）：{sub_path.name}")
        return

    # targets -> output dirs
    out_dirs = {}
    for t in targets:
        out_dir = sub_path / f"{t}_{t}b_downsample"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[t] = out_dir

    # counters
    ok = {t: 0 for t in targets}

    # 可视化输出目录
    viz_dir = sub_path / "viz_check"
    if viz_save:
        viz_dir.mkdir(parents=True, exist_ok=True)

    files_128b = sorted(dir_128b.glob("*.bin"))
    if not files_128b:
        print(f"⚠️ 跳过（ouster128b_label无bin）：{sub_path.name}")
        return

    # 抽样帧（从 128b 的帧集合中抽）
    sample_files = set()
    if do_viz and viz_samples > 0:
        k = min(viz_samples, len(files_128b))
        sample_files = set(random.sample(files_128b, k))

    missing_128 = 0

    print(f"\n📂 处理：{sub_path.name}")
    print(f"  - 输入：ouster128b_label + ouster128_label")
    print(f"  - 输出：{', '.join([f'{t}_{t}b_downsample' for t in targets])}")
    print(f"  - 下采样：num_lines={num_lines}, fov=[{fov_down},{fov_up}], start_offset={start_offset}")

    for f128b in files_128b:
        name = f128b.name
        f128 = dir_128 / name

        try:
            pc_128b = read_bin_kitti(str(f128b), 4)
        except Exception as e:
            print(f"❌ 读取失败：{f128b} -> {e}")
            continue

        pc_128 = None
        if f128.exists():
            try:
                pc_128 = read_bin_kitti(str(f128), 4)
            except Exception as e:
                print(f"❌ 读取失败：{f128} -> {e}")
                pc_128 = None
        else:
            missing_128 += 1

        # 原始合并（用于可视化对比）
        merged_128 = merge_two_clouds(pc_128b, pc_128)

        merged_by_t = {}  # for optional viz

        for t in targets:
            try:
                pc_128b_t = downsample_vertical(pc_128b, num_lines, t, fov_down, fov_up, start_offset)
                pc_128_t  = downsample_vertical(pc_128,  num_lines, t, fov_down, fov_up, start_offset) if pc_128 is not None else None
                merged_t = merge_two_clouds(pc_128b_t, pc_128_t)

                write_bin_kitti(str(out_dirs[t] / name), merged_t)
                ok[t] += 1
                merged_by_t[t] = merged_t
            except Exception as e:
                print(f"❌ {t}处理失败：{name} -> {e}")
                # 这里选择 continue（跳过该 t），而不是整帧失败
                continue

        # ====== 可视化抽查 + 保存统计图 ======
        if f128b in sample_files:
            base = os.path.splitext(name)[0]
            title = f"{sub_path.name} | {base} (red=128 merge, green=64 merge, blue=16 merge)"

            if viz_save:
                # 保存 ring/elev 统计：对 merged_128 保存（也可改为分别保存 128b/128）
                prefix = str(viz_dir / f"{base}_merged128")
                p1, p2 = save_ring_stats_png(merged_128, num_lines, fov_down, fov_up, prefix)
                print(f"🖼️ 保存统计图：{p1}, {p2}")

            if do_viz:
                visualize_overlay_triplet(
                    merged_128=merged_128,
                    merged_64=merged_64,
                    merged_16=merged_16,
                    title=title,
                    max_points=viz_max_points
                )

    print(f"✅ 完成：{sub_path.name}")
    for t in targets:
        print(f"  - {t}输出：{ok[t]} 帧")
    if missing_128 > 0:
        print(f"  - 缺少 ouster128_label 对应帧：{missing_128}（该帧只用128b下采样参与合并）")


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser("Downsample ouster128b_label & ouster128_label to 64/16 and merge; with visual checks.")
    ap.add_argument("--root_dir", type=str, default="/home/suolab/OpenPCDet/data/sunlakes/2026-01-04_18_02_01-export/resource", help="Root dir containing rosbag2_2026_* folders")
    ap.add_argument("--sub_prefix", type=str, default="rosbag2_", help="Subdir prefix")

    ap.add_argument("--num_lines", type=int, default=128, help="Assumed original vertical beams")
    ap.add_argument("--fov_down", type=float, default=-22.5, help="Vertical FOV lower bound (deg)")
    ap.add_argument("--fov_up", type=float, default=22.5, help="Vertical FOV upper bound (deg)")
    ap.add_argument("--start_offset", type=int, default=0, help="Ring keep offset within step (0..step-1)")

    ap.add_argument("--viz", action="store_true", help="Open3D overlay visualization on sampled frames")
    ap.add_argument("--viz_samples", type=int, default=5, help="How many frames to visualize per subdir (random)")
    ap.add_argument("--viz_save_png", action="store_true", help="Save ring/elev PNG plots to subdir/viz_check/")
    ap.add_argument("--viz_max_points", type=int, default=250000, help="Max points per cloud in Open3D (random downsample)")
    
    ap.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[16],
        help="Target vertical lines to generate, e.g. --targets 64 32 16"
    )
    args = ap.parse_args()

    root = Path(args.root_dir)
    assert root.exists(), f"root_dir not found: {root}"

    subs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(args.sub_prefix)]
    if not subs:
        print(f"⚠️ 未找到子目录：{args.sub_prefix}* in {root}")
        return

    for sub in sorted(subs):
        process_one_subdir(
            sub_path=sub,
            num_lines=args.num_lines,
            fov_down=args.fov_down,
            fov_up=args.fov_up,
            start_offset=args.start_offset,
            targets=args.targets,
            do_viz=args.viz,
            viz_samples=args.viz_samples,
            viz_save=args.viz_save_png,
            viz_max_points=args.viz_max_points
        )

    print("\n🎉 全部完成")


if __name__ == "__main__":
    main()
