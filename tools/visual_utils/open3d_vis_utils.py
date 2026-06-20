"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.

Modified:
- color 3D boxes by class name using a global colormap (tab20),
  matching OpenPCDet model label order (CLASS_NAMES)
- support intensity-colored point cloud
- support static horizontal / vertical line
- support interactive keyboard-controlled line editing
"""

import open3d as o3d
import torch
import matplotlib
import numpy as np

from typing import List, Dict
from matplotlib.cm import get_cmap


# ============================================================
# MUST match the model output label order:
# 1->car, 2->truck, 3->construction_vehicle, ...
# ============================================================
MODEL_CLASS_NAMES: List[str] = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
PALETTE = "tab20"

COLOR_CLASS_ORDER: List[str] = [
    'car', 'truck', 'trailer', 'pedestrian', 'barrier',
    'motorcycle', 'bus', 'construction_vehicle', 'bicycle', 'traffic_cone'
]


def make_global_color_map(class_order: List[str], palette_name: str = PALETTE) -> Dict[str, tuple]:
    cmap = get_cmap(palette_name)
    return {c: cmap(i % cmap.N) for i, c in enumerate(class_order)}  # RGBA in [0,1]


GLOBAL_COLOR_MAP = make_global_color_map(COLOR_CLASS_ORDER)


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def label_to_rgb(label_id: int, labels_base_is_one: bool, default=(1.0, 1.0, 1.0)) -> List[float]:
    lid = int(label_id)
    idx = lid - 1 if labels_base_is_one else lid

    if idx < 0 or idx >= len(MODEL_CLASS_NAMES):
        return list(default)

    model_cls_name = MODEL_CLASS_NAMES[idx]
    rgba = GLOBAL_COLOR_MAP.get(model_cls_name, None)
    if rgba is None:
        return list(default)

    rgb = np.array(rgba[:3], dtype=np.float32)
    return np.clip(rgb, 0.0, 1.0).tolist()


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num + 1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def create_axis_line(mode='horizontal',
                     x1=-20.0, x2=20.0,
                     y1=-20.0, y2=20.0,
                     x=0.0, y=0.0, z=0.0,
                     color=(1.0, 0.0, 0.0)):
    """
    mode='horizontal': line along x-axis, fixed y and z
        P1 = (x1, y, z)
        P2 = (x2, y, z)

    mode='vertical': line along y-axis, fixed x and z
        P1 = (x, y1, z)
        P2 = (x, y2, z)
    """
    if mode == 'horizontal':
        points = np.array([
            [x1, y, z],
            [x2, y, z]
        ], dtype=np.float64)
    elif mode == 'vertical':
        points = np.array([
            [x, y1, z],
            [x, y2, z]
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported mode: {mode}, expected 'horizontal' or 'vertical'")

    lines = np.array([[0, 1]], dtype=np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(np.array([color], dtype=np.float64))
    return line_set


def create_horizontal_line(x1=-20.0, x2=20.0, y=0.0, z=0.0, color=(1.0, 0.0, 0.0)):
    return create_axis_line(
        mode='horizontal',
        x1=x1, x2=x2,
        y=y, z=z,
        color=color
    )


def create_vertical_line(y1=-20.0, y2=20.0, x=0.0, z=0.0, color=(1.0, 0.0, 0.0)):
    return create_axis_line(
        mode='vertical',
        y1=y1, y2=y2,
        x=x, z=z,
        color=color
    )


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None,
                point_colors=None, pcd_color=(1.0, 1.0, 1.0), extra_geometries=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.detach().cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.detach().cpu().numpy()
    if isinstance(ref_labels, torch.Tensor):
        ref_labels = ref_labels.detach().cpu().numpy()

    geometries = []

    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if point_colors is not None:
        point_colors = np.asarray(point_colors, dtype=np.float32)
        point_colors = np.clip(point_colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        c = np.array(pcd_color, dtype=np.float32)
        c = np.clip(c, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (points.shape[0], 1)))
    geometries.append(pcd)

    # coordinate frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]))

    # gt boxes
    if gt_boxes is not None:
        for i in range(gt_boxes.shape[0]):
            line_set, _ = translate_boxes_to_open3d_instance(gt_boxes[i])
            line_set.paint_uniform_color([0, 0, 1])
            geometries.append(line_set)

    # predicted boxes
    if ref_boxes is not None:
        labels_base_is_one = True
        if ref_labels is not None and len(ref_labels) > 0:
            labels_base_is_one = (np.min(ref_labels) >= 1)

        for i in range(ref_boxes.shape[0]):
            line_set, _ = translate_boxes_to_open3d_instance(ref_boxes[i])
            if ref_labels is None:
                c = [0, 1, 0]
            else:
                c = label_to_rgb(ref_labels[i], labels_base_is_one, default=(1, 1, 1))
            line_set.paint_uniform_color(c)
            geometries.append(line_set)

    if extra_geometries is not None:
        geometries.extend(extra_geometries)

    o3d.visualization.draw(
        geometries,
        bg_color=(0, 0, 0, 1),
        point_size=1,
        line_width=5,
        show_ui=True
    )


def interactive_draw_scenes_with_line(points, gt_boxes=None, ref_boxes=None, ref_labels=None,
                                      point_colors=None, pcd_color=(1.0, 1.0, 1.0),
                                      line_mode='horizontal',
                                      x1=-100.0, x2=100.0,
                                      y1=-100.0, y2=100.0,
                                      x=0.0, y=5.0, z=0.0,
                                      line_color=(1.0, 0.0, 0.0), step=0.5,
                                      endpoint_radius=0.5):
    """
    Keyboard interactive line editor.

    line_mode:
        'horizontal' : along x, fixed y/z
        'vertical'   : along y, fixed x/z

    Controls:
        T     : toggle horizontal / vertical
        A / D : move endpoint 1 along line axis
        J / L : move endpoint 2 along line axis
        W / S : move fixed axis
        Q / E : move z
        R     : reset
        ESC   : close window
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.detach().cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.detach().cpu().numpy()
    if isinstance(ref_labels, torch.Tensor):
        ref_labels = ref_labels.detach().cpu().numpy()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Open3D Interactive Line Editor', width=1600, height=900)

    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    render_opt.point_size = 1.0
    render_opt.line_width = 5.0

    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if point_colors is not None:
        point_colors = np.asarray(point_colors, dtype=np.float32)
        point_colors = np.clip(point_colors, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
    else:
        c = np.array(pcd_color, dtype=np.float32)
        c = np.clip(c, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (points.shape[0], 1)))
    vis.add_geometry(pcd)

    # coordinate frame
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # gt boxes
    if gt_boxes is not None:
        for i in range(gt_boxes.shape[0]):
            line_set, _ = translate_boxes_to_open3d_instance(gt_boxes[i])
            line_set.paint_uniform_color([0, 0, 1])
            vis.add_geometry(line_set)

    # predicted boxes
    if ref_boxes is not None:
        labels_base_is_one = True
        if ref_labels is not None and len(ref_labels) > 0:
            labels_base_is_one = (np.min(ref_labels) >= 1)

        for i in range(ref_boxes.shape[0]):
            line_set, _ = translate_boxes_to_open3d_instance(ref_boxes[i])
            if ref_labels is None:
                c = [0, 1, 0]
            else:
                c = label_to_rgb(ref_labels[i], labels_base_is_one, default=(1, 1, 1))
            line_set.paint_uniform_color(c)
            vis.add_geometry(line_set)

    state = {
        'mode': str(line_mode),
        'x1': float(x1),
        'x2': float(x2),
        'y1': float(y1),
        'y2': float(y2),
        'x': float(x),
        'y': float(y),
        'z': float(z),
        'line': None,
        'p1_sphere': None,
        'p2_sphere': None,
    }

    init_state = {
        'mode': str(line_mode),
        'x1': float(x1),
        'x2': float(x2),
        'y1': float(y1),
        'y2': float(y2),
        'x': float(x),
        'y': float(y),
        'z': float(z),
    }

    def get_endpoints():
        if state['mode'] == 'horizontal':
            p1 = np.array([state['x1'], state['y'], state['z']], dtype=np.float64)
            p2 = np.array([state['x2'], state['y'], state['z']], dtype=np.float64)
        elif state['mode'] == 'vertical':
            p1 = np.array([state['x'], state['y1'], state['z']], dtype=np.float64)
            p2 = np.array([state['x'], state['y2'], state['z']], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported line mode: {state['mode']}")
        return p1, p2

    def make_line_and_points():
        p1_xyz, p2_xyz = get_endpoints()
        pts = np.stack([p1_xyz, p2_xyz], axis=0)

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(pts)
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
        line.colors = o3d.utility.Vector3dVector(np.array([line_color], dtype=np.float64))

        p1_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=endpoint_radius)
        p1_mesh.translate(p1_xyz)
        p1_mesh.paint_uniform_color([1.0, 1.0, 0.0])  # yellow

        p2_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=endpoint_radius)
        p2_mesh.translate(p2_xyz)
        p2_mesh.paint_uniform_color([0.0, 1.0, 1.0])  # cyan

        return line, p1_mesh, p2_mesh

    def print_state():
        p1_xyz, p2_xyz = get_endpoints()
        length = np.linalg.norm(p2_xyz - p1_xyz)
        print(
            f"\rmode={state['mode']}   "
            f"P1=({p1_xyz[0]:.2f}, {p1_xyz[1]:.2f}, {p1_xyz[2]:.2f})   "
            f"P2=({p2_xyz[0]:.2f}, {p2_xyz[1]:.2f}, {p2_xyz[2]:.2f})   "
            f"Length={length:.2f} m",
            end='',
            flush=True
        )

    def refresh_line(vis_):
        if state['line'] is not None:
            vis_.remove_geometry(state['line'], reset_bounding_box=False)
        if state['p1_sphere'] is not None:
            vis_.remove_geometry(state['p1_sphere'], reset_bounding_box=False)
        if state['p2_sphere'] is not None:
            vis_.remove_geometry(state['p2_sphere'], reset_bounding_box=False)

        state['line'], state['p1_sphere'], state['p2_sphere'] = make_line_and_points()

        vis_.add_geometry(state['line'], reset_bounding_box=False)
        vis_.add_geometry(state['p1_sphere'], reset_bounding_box=False)
        vis_.add_geometry(state['p2_sphere'], reset_bounding_box=False)

        vis_.update_geometry(state['line'])
        vis_.update_geometry(state['p1_sphere'])
        vis_.update_geometry(state['p2_sphere'])
        vis_.poll_events()
        vis_.update_renderer()

        print_state()
        return False

    def move_p1_neg(vis_):
        if state['mode'] == 'horizontal':
            state['x1'] -= step
        else:
            state['y1'] -= step
        return refresh_line(vis_)

    def move_p1_pos(vis_):
        if state['mode'] == 'horizontal':
            state['x1'] += step
        else:
            state['y1'] += step
        return refresh_line(vis_)

    def move_p2_neg(vis_):
        if state['mode'] == 'horizontal':
            state['x2'] -= step
        else:
            state['y2'] -= step
        return refresh_line(vis_)

    def move_p2_pos(vis_):
        if state['mode'] == 'horizontal':
            state['x2'] += step
        else:
            state['y2'] += step
        return refresh_line(vis_)

    def move_fixed_axis_pos(vis_):
        if state['mode'] == 'horizontal':
            state['y'] += step
        else:
            state['x'] += step
        return refresh_line(vis_)

    def move_fixed_axis_neg(vis_):
        if state['mode'] == 'horizontal':
            state['y'] -= step
        else:
            state['x'] -= step
        return refresh_line(vis_)

    def move_z_up(vis_):
        state['z'] += step
        return refresh_line(vis_)

    def move_z_down(vis_):
        state['z'] -= step
        return refresh_line(vis_)

    def toggle_mode(vis_):
        if state['mode'] == 'horizontal':
            state['mode'] = 'vertical'
        else:
            state['mode'] = 'horizontal'
        return refresh_line(vis_)

    def reset_line(vis_):
        state['mode'] = init_state['mode']
        state['x1'] = init_state['x1']
        state['x2'] = init_state['x2']
        state['y1'] = init_state['y1']
        state['y2'] = init_state['y2']
        state['x'] = init_state['x']
        state['y'] = init_state['y']
        state['z'] = init_state['z']
        return refresh_line(vis_)

    refresh_line(vis)

    vis.register_key_callback(ord('T'), toggle_mode)
    vis.register_key_callback(ord('A'), move_p1_neg)
    vis.register_key_callback(ord('D'), move_p1_pos)
    vis.register_key_callback(ord('J'), move_p2_neg)
    vis.register_key_callback(ord('L'), move_p2_pos)
    vis.register_key_callback(ord('W'), move_fixed_axis_pos)
    vis.register_key_callback(ord('S'), move_fixed_axis_neg)
    vis.register_key_callback(ord('Q'), move_z_up)
    vis.register_key_callback(ord('E'), move_z_down)
    vis.register_key_callback(ord('R'), reset_line)

    print("\nKeyboard controls:")
    print("  T     : toggle horizontal / vertical")
    print("  A / D : move endpoint 1 along line axis")
    print("  J / L : move endpoint 2 along line axis")
    print("  W / S : move fixed axis")
    print("  Q / E : move line along z")
    print("  R     : reset line")
    print("  ESC   : close window")

    vis.run()
    vis.destroy_window()
    print()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    """
    If ref_labels is provided, color each box by CLASS_NAMES/GLOBAL_COLOR_MAP.
    Handles both 1-based and 0-based label ids automatically.
    """
    labels_base_is_one = True
    if ref_labels is not None:
        ref_labels_np = _to_numpy(ref_labels).astype(np.int64).reshape(-1)
        if ref_labels_np.size > 0:
            labels_base_is_one = (ref_labels_np.min() >= 1)

    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])

        if ref_labels is None:
            c = color
        else:
            c = label_to_rgb(ref_labels[i], labels_base_is_one, default=(1.0, 1.0, 1.0))

        c = np.array(c, dtype=np.float32)
        c = np.clip(c, 0.0, 1.0).tolist()

        line_set.paint_uniform_color(c)
        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])

    return vis
