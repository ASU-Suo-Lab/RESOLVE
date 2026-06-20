#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d


def read_bin_points(bin_path: str, dim: int = 5) -> np.ndarray:
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % dim != 0:
        raise ValueError(f"File size not divisible by dim={dim}. floats={arr.size}")
    return arr.reshape(-1, dim)


def find_box_from_dbinfos(bin_path: str, dbinfos_pkl: str):
    bin_path = Path(bin_path).resolve()
    data_root = Path(dbinfos_pkl).resolve().parent

    rel = None
    if str(bin_path).startswith(str(data_root)):
        rel = bin_path.relative_to(data_root).as_posix()

    with open(dbinfos_pkl, "rb") as f:
        dbinfos = pickle.load(f)

    for cls_name, items in dbinfos.items():
        for it in items:
            p = it.get("path", None)
            if p is None:
                continue
            p_norm = p.replace("\\", "/")
            if rel is not None and p_norm == rel:
                return np.array(it["box3d_lidar"], dtype=np.float32), it.get("name", cls_name)
            if Path(p_norm).name == bin_path.name:
                return np.array(it["box3d_lidar"], dtype=np.float32), it.get("name", cls_name)

    return None, None


def make_pcd(points_xyz: np.ndarray, rgb: tuple[float, float, float], max_points: int = 200000):
    if points_xyz.shape[0] == 0:
        return None
    pts = points_xyz.astype(np.float64)
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    colors = np.tile(np.array(rgb, dtype=np.float64)[None, :], (pts.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def make_box_lines_local(box3d_lidar: np.ndarray | None, color=(0.0, 0.0, 0.0)):
    """
    默认：GT bin 是局部坐标（已减中心），所以 box 画在原点。
    若你的 bin 是全局坐标，把 center 改成 box3d_lidar[:3]。
    """
    if box3d_lidar is None:
        return None

    dx, dy, dz = float(box3d_lidar[3]), float(box3d_lidar[4]), float(box3d_lidar[5])
    yaw = float(box3d_lidar[6])

    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, yaw])
    obb = o3d.geometry.OrientedBoundingBox(
        center=np.array([0.0, 0.0, 0.0]),  # 局部坐标
        R=R,
        extent=np.array([dx, dy, dz], dtype=np.float64),
    )
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lines.paint_uniform_color(list(color))
    return lines


def main():
    ap = argparse.ArgumentParser("Manual view -> save 3 images (High red / Mid green / Low blue, box black)")
    ap.add_argument("--gt_high", type=str, default="/home/suolab/OpenPCDet/data/sunlakes/high_2lidar_cam/v1.0-trainval/sunlakes_gt_database/2486_bicycle_4.bin")
    ap.add_argument("--gt_mid", type=str, default="/home/suolab/OpenPCDet/data/sunlakes/mid_2lidar_cam/v1.0-trainval/sunlakes_gt_database/2486_bicycle_4.bin")
    ap.add_argument("--gt_low", type=str, default="/home/suolab/OpenPCDet/data/sunlakes/low_2lidar_cam/v1.0-trainval/sunlakes_gt_database/2486_bicycle_5.bin")
    ap.add_argument("--dim", type=int, default=5)
    ap.add_argument("--dbinfos", type=str, default="/home/suolab/OpenPCDet/data/sunlakes/high_2lidar_cam/v1.0-trainval/sunlakes_dbinfos_train.pkl")
    ap.add_argument("--out_dir", type=str, default="./o3d_renders")
    ap.add_argument("--max_points", type=int, default=200000)
    ap.add_argument("--point_size", type=float, default=5.0)  # 仅对旧 Visualizer 有效（draw_geometries 没法）
    args = ap.parse_args()

    for p in [args.gt_high, args.gt_mid, args.gt_low]:
        assert os.path.exists(p), f"Not found: {p}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load point clouds ---
    high_xyz = read_bin_points(args.gt_high, args.dim)[:, :3]
    mid_xyz  = read_bin_points(args.gt_mid,  args.dim)[:, :3]
    low_xyz  = read_bin_points(args.gt_low,  args.dim)[:, :3]

    pcd_high = make_pcd(high_xyz, (1.0, 0.0, 0.0), args.max_points)  # red
    pcd_mid  = make_pcd(mid_xyz,  (0.0, 1.0, 0.0), args.max_points)  # green
    pcd_low  = make_pcd(low_xyz,  (0.0, 0.0, 1.0), args.max_points)  # blue

    # --- load box (optional) ---
    box = None
    name = None
    if args.dbinfos is not None:
        assert os.path.exists(args.dbinfos), f"Not found: {args.dbinfos}"
        box, name = find_box_from_dbinfos(args.gt_high, args.dbinfos)
        if box is None:
            print(f"[Warn] Box not found in dbinfos for {args.gt_high}. Will render without box.")
    
    #box_lines = make_box_lines_local(box, color=(0.0, 0.0, 0.0))  # black
    box_lines =None
    
    stem = Path(args.gt_high).stem
    # --- compute distance to reference (-22, 0) using dbinfos box center ---
    ref_xy = np.array([23.0, 18.0], dtype=np.float32)
    dist_tag = "" 
    if box is not None:
        gt_xy = box[:2].astype(np.float32)  # (cx, cy)
        dist = float(np.linalg.norm(gt_xy - ref_xy))
        dist_tag = f"_d{dist:.2f}m"          # 例如 _d13.57m
        print(f"[Dist] GT center=({gt_xy[0]:.3f},{gt_xy[1]:.3f}) -> (23.0, 18.0) = {dist:.3f} m")
    else:
        print("[Warn] box is None, distance tag will be omitted.")

    name_tag = f"_{name}" if name else ""

    out_high = str(out_dir / f"{stem}{name_tag}{dist_tag}_HIGH_red.jpg")
    out_mid  = str(out_dir / f"{stem}{name_tag}{dist_tag}_MID_green.jpg")
    out_low  = str(out_dir / f"{stem}{name_tag}{dist_tag}_LOW_blue.jpg")

    # --- interactive visualizer with key callbacks ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Adjust view, press S to save 3 images | 1/2/3 switch", width=1280, height=720)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.asarray([1.0, 1.0, 1.0])  # white background (good for black box)
    render_opt.point_size = float(args.point_size)

    state = {"mode": "high"}  # current shown

    def clear_and_show(which: str):
        vis.clear_geometries()
        if which == "high":
            vis.add_geometry(pcd_high, reset_bounding_box=False)
        elif which == "mid":
            vis.add_geometry(pcd_mid, reset_bounding_box=False)
        elif which == "low":
            vis.add_geometry(pcd_low, reset_bounding_box=False)
        else:
            raise ValueError(which)

        if box_lines is not None:
            vis.add_geometry(box_lines, reset_bounding_box=False)

        state["mode"] = which
        vis.poll_events()
        vis.update_renderer()

    # 先显示 high，并 reset bbox 一次（方便初始视角）
    vis.add_geometry(pcd_high, reset_bounding_box=True)
    if box_lines is not None:
        vis.add_geometry(box_lines, reset_bounding_box=False)

    # --- key callbacks ---
    def on_1(v):
        clear_and_show("high")
        print("[View] show HIGH (red)")
        return False

    def on_2(v):
        clear_and_show("mid")
        print("[View] show MID (green)")
        return False

    def on_3(v):
        clear_and_show("low")
        print("[View] show LOW (blue)")
        return False

    def on_s(v):
        vc = v.get_view_control()
        cam = vc.convert_to_pinhole_camera_parameters()

        def snap_new_window(pcd, out_path: str):
            vis2 = o3d.visualization.Visualizer()
            vis2.create_window(visible=False, width=1280, height=720)

            opt = vis2.get_render_option()
            opt.background_color = np.asarray([1.0, 1.0, 1.0])
            opt.point_size = float(args.point_size)

            vis2.add_geometry(pcd, reset_bounding_box=True)
            if box_lines is not None:
                vis2.add_geometry(box_lines, reset_bounding_box=False)

            # 应用同一视角
            vc2 = vis2.get_view_control()
            vc2.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

            vis2.poll_events()
            vis2.update_renderer()
            vis2.capture_screen_image(out_path, do_render=True)
            vis2.destroy_window()
            print(f"[Save] {out_path}")

        snap_new_window(pcd_high, out_high)
        snap_new_window(pcd_mid,  out_mid)
        snap_new_window(pcd_low,  out_low)

        print("[OK] saved 3 images with the same manual view.")
        return False

    vis.register_key_callback(ord("1"), on_1)
    vis.register_key_callback(ord("2"), on_2)
    vis.register_key_callback(ord("3"), on_3)
    vis.register_key_callback(ord("S"), on_s)
    vis.register_key_callback(ord("s"), on_s)

    print("Controls:")
    print("  - Mouse: rotate/pan/zoom to set view")
    print("  - Key 1/2/3: show HIGH/MID/LOW")
    print("  - Key S: save 3 images (HIGH red, MID green, LOW blue), box black")
    print("  - Close window or press Q to exit (depends on OS)")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
