import numpy as np
import os
import numba
import json
from tqdm import tqdm
from typing import Tuple, List, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import open3d as o3d
import random
import re
import cv2
from shapely.geometry import MultiPoint, box

INTRINSIC_PAT = re.compile(r"Intrinsic:\s*\n([^\n]+)\n([^\n]+)\n([^\n]+)")
DISTORT_PAT = re.compile(r"Distortion:\s*\n\[(.*?)\]", re.S)
R_PAT = re.compile(r"R:\s*(.*?)\s*t:", re.S)
T_PAT = re.compile(r"t:\s*(.*?)\n")

_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

def read_points(file_path, dim=4):
    suffix = os.path.splitext(file_path)[1]
    assert suffix in ['.bin', '.ply']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    else:
        raise NotImplementedError


def get_extrinsic(calib_path: Path) -> np.ndarray:
    content = Path(calib_path).read_text(encoding="utf-8")

    mR = R_PAT.search(content)
    if not mR:
        raise ValueError(f"未找到 R 矩阵: {calib_path}")
    R_lines = [list(map(float, line.strip().split())) for line in mR.group(1).strip().splitlines()]

    mT = T_PAT.search(content)
    if not mT:
        raise ValueError(f"未找到 t 向量: {calib_path}")
    t_vec = list(map(float, mT.group(1).strip().split()))

    extr = np.eye(4, dtype=np.float32)
    extr[:3, :3] = np.asarray(R_lines, dtype=np.float32)
    extr[:3, 3] = np.asarray(t_vec, dtype=np.float32)
    return extr


def get_cam_kd(calib_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    content = Path(calib_path).read_text(encoding="utf-8")
    m = INTRINSIC_PAT.search(content)
    if m is None:
        raise ValueError(f"Intrinsic 未找到: {calib_path}")
    K = np.vstack([np.fromstring(row, sep=" ") for row in m.groups()]).astype(np.float32)

    m = DISTORT_PAT.search(content)
    D = np.fromstring(m.group(1), sep=",", dtype=np.float32) if m else np.zeros(5, np.float32)
    return K, D


def adjust_label_segment(pc_rel_path: str, sensor: str) -> str:
    """
    将相对路径中的 '128_128b_label' / '64_64b_label' / '16_16b_label'
    统一替换为与 sensor 对应的目录名。
    """
    target = {
        'os128': '128_128b_label',
        'os64': '64_64b_label',
        'rs16': '16_16b_label',
        'rs16down': '16_16b_downsample',
        'os64down':'64_64b_downsample'
    }.get(sensor, '128_128b_label')

    p = Path(pc_rel_path)
    parts = list(p.parts)
    for i, seg in enumerate(parts):
        if seg.endswith('_label') and seg.split('_')[0] in {'128', '64', '16'}:
            parts[i] = target
    # 如果路径里没有 *_label 段，但你仍想强制加入，按需在此添加逻辑；目前仅替换存在的段
    return str(Path(*parts))


def pc_id_to_ts(pc_id: str) -> float:
    sec, nano = map(int, pc_id.split('-'))
    return sec + nano / 1e9


def rotmat_z(theta_deg: float, homogeneous: bool = False) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    if homogeneous:
        R = np.eye(4, dtype=np.float32)
        R[:3, :3] = np.array([[c, -s, 0.0],
                              [s, c, 0.0],
                              [0.0, 0.0, 1.0]], dtype=np.float32)
        return R
    else:
        return np.array([[c, -s, 0.0],
                         [s, c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float32)


def crop_img_list(img_list, boxes):
    img_patches = []
    for i in range(boxes.shape[0]):
        bbox = boxes[i, :].astype(np.int32)
        x1, y1, x2, y2, num_img = bbox
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)

        img_patch = img_list[num_img][y1:y1 + h, x1:x1 + w]
        img_patches.append(img_patch)
    return img_patches


def post_process_coords(
        corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def corners_for_projection_from_gtboxes(gt_boxes_full_7, yaw_flip=False):
    """
    gt_boxes_full_7: (N,7) [x,y,z,dx,dy,dz,yaw] 其中 z 是中心高度
    返回: (N,3,8) corners in lidar frame
    """
    b = gt_boxes_full_7.copy().astype(np.float32)

    # 把 center.z 转成 bottom.z，匹配 bbox3d2corners 的 z=0..dz 模板
    b[:, 2] -= b[:, 5] / 2.0

    if yaw_flip:
        b[:, 6] = -b[:, 6]

    corners = bbox3d2corners(b)  # (N,8,3)
    corners = corners.transpose(0, 2, 1)  # (N,3,8)
    return corners


def bbox3d2corners(bboxes):
    '''
    bboxes: shape=(n, 7)
    return: shape=(n, 8, 3)
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |
    y      | /              |  |     | |
    <------|o               | 7 -----| 4
                            |/   o   |/
                            3 ------ 0
    x: front, y: left, z: top
    '''
    # TODO DSZ 这里yaw要取反
    centers, dims, angles = bboxes[:, :3], bboxes[:, 3:6], -bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bboxes_corners = np.array([[-0.5, -0.5, 0], [-0.5, -0.5, 1.0], [-0.5, 0.5, 1.0], [-0.5, 0.5, 0.0],
                               [0.5, -0.5, 0], [0.5, -0.5, 1.0], [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]],
                              dtype=np.float32)
    bboxes_corners = bboxes_corners[None, :, :] * dims[:, None, :]  # (1, 8, 3) * (n, 1, 3) -> (n, 8, 3)

    # 2. rotate around z axis
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    # in fact, -angle
    rot_mat = np.array([[rot_cos, rot_sin, np.zeros_like(rot_cos)],
                        [-rot_sin, rot_cos, np.zeros_like(rot_cos)],
                        [np.zeros_like(rot_cos), np.zeros_like(rot_cos), np.ones_like(rot_cos)]],
                       dtype=np.float32)  # (3, 3, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))  # (n, 3, 3)
    bboxes_corners = bboxes_corners @ rot_mat  # (n, 8, 3)

    # 3. translate to centers
    bboxes_corners += centers[:, None, :]
    return bboxes_corners


def get_points_num_in_bbox(points, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3)
    location: shape=(n, 3)
    rotation_y: shape=(n, )
    name: shape=(n, )
    return: shape=(n, )
    '''
    indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name = \
        points_in_bboxes_v2(
            points=points,
            dimensions=dimensions,
            location=location,
            rotation_y=rotation_y,
            name=name)
    points_num = np.sum(indices, axis=0)
    non_valid_points_num = [-1] * (n_total_bbox - n_valid_bbox)
    points_num = np.concatenate([points_num, non_valid_points_num], axis=0)
    return np.array(points_num, dtype=np.int32)


def points_in_bboxes_v2(points, dimensions, location, rotation_y, name):
    '''
    points: shape=(N, 4)
    tr_velo_to_cam: shape=(4, 4)
    r0_rect: shape=(4, 4)
    dimensions: shape=(n, 3)
    location: shape=(n, 3)
    rotation_y: shape=(n, )
    name: shape=(n, )
    return:
        indices: shape=(N, n_valid_bbox), indices[i, j] denotes whether point i is in bbox j.
        n_total_bbox: int.
        n_valid_bbox: int, not including 'DontCare'
        bboxes_lidar: shape=(n_valid_bbox, 7)
        name: shape=(n_valid_bbox, )
    '''
    n_total_bbox = len(dimensions)
    n_valid_bbox = len([item for item in name if item != 'DontCare'])
    location, dimensions = location[:n_valid_bbox], dimensions[:n_valid_bbox]
    rotation_y, name = rotation_y[:n_valid_bbox], name[:n_valid_bbox]
    bboxes_lidar = np.concatenate([location, dimensions, rotation_y[:, None]], axis=1)
    bboxes_corners = bbox3d2corners(bboxes_lidar)
    group_rectangle_vertexs_v = group_rectangle_vertexs(bboxes_corners)
    frustum_surfaces = group_plane_equation(group_rectangle_vertexs_v)
    indices = points_in_bboxes(points[:, :3], frustum_surfaces)  # (N, n), N is points num, n is bboxes number
    return indices, n_total_bbox, n_valid_bbox, bboxes_lidar, name


def group_rectangle_vertexs(bboxes_corners):
    '''
    bboxes_corners: shape=(n, 8, 3)
    return: shape=(n, 6, 4, 3)
    '''
    rec1 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 1], bboxes_corners[:, 3], bboxes_corners[:, 2]],
                    axis=1)  # (n, 4, 3)
    rec2 = np.stack([bboxes_corners[:, 4], bboxes_corners[:, 7], bboxes_corners[:, 6], bboxes_corners[:, 5]],
                    axis=1)  # (n, 4, 3)
    rec3 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 4], bboxes_corners[:, 5], bboxes_corners[:, 1]],
                    axis=1)  # (n, 4, 3)
    rec4 = np.stack([bboxes_corners[:, 2], bboxes_corners[:, 6], bboxes_corners[:, 7], bboxes_corners[:, 3]],
                    axis=1)  # (n, 4, 3)
    rec5 = np.stack([bboxes_corners[:, 1], bboxes_corners[:, 5], bboxes_corners[:, 6], bboxes_corners[:, 2]],
                    axis=1)  # (n, 4, 3)
    rec6 = np.stack([bboxes_corners[:, 0], bboxes_corners[:, 3], bboxes_corners[:, 7], bboxes_corners[:, 4]],
                    axis=1)  # (n, 4, 3)
    group_rectangle_vertexs = np.stack([rec1, rec2, rec3, rec4, rec5, rec6], axis=1)
    return group_rectangle_vertexs


def group_plane_equation(bbox_group_rectangle_vertexs):
    '''
    bbox_group_rectangle_vertexs: shape=(n, 6, 4, 3)
    return: shape=(n, 6, 4)
    '''
    # 1. generate vectors for a x b
    vectors = bbox_group_rectangle_vertexs[:, :, :2] - bbox_group_rectangle_vertexs[:, :, 1:3]
    normal_vectors = np.cross(vectors[:, :, 0], vectors[:, :, 1])  # (n, 6, 3)
    normal_d = np.einsum('ijk,ijk->ij', bbox_group_rectangle_vertexs[:, :, 0], normal_vectors)  # (n, 6)
    plane_equation_params = np.concatenate([normal_vectors, -normal_d[:, :, None]], axis=-1)
    return plane_equation_params


@numba.jit(nopython=True)
def points_in_bboxes(points, plane_equation_params):
    '''
    points: shape=(N, 3)
    plane_equation_params: shape=(n, 6, 4)
    return: shape=(N, n), bool
    '''
    N, n = len(points), len(plane_equation_params)
    m = plane_equation_params.shape[1]
    masks = np.ones((N, n), dtype=np.bool_)
    for i in range(N):
        x, y, z = points[i, :3]
        for j in range(n):
            bbox_plane_equation_params = plane_equation_params[j]
            for k in range(m):
                a, b, c, d = bbox_plane_equation_params[k]
                if a * x + b * y + c * z + d >= 0:
                    masks[i][j] = False
                    break
    return masks


def _rotate_one_frame(args_tuple):
    data_root, data_prefix, sensor_type, rot_pc, frame = args_tuple
    data_root = Path(data_root)
    json_pc_rel_path = '/'.join(frame['info'].split('/')[-3:])
    pc_rel_path = adjust_label_segment(json_pc_rel_path, sensor_type)

    resource_root = data_root / data_prefix / 'resource'
    lidar_path = resource_root / pc_rel_path
    if not os.path.exists(lidar_path):
        return ('skip', str(lidar_path))  # 用于统计/日志

    pc_id = Path(pc_rel_path).stem
    sec, nano = map(int, pc_id.split('-'))
    ts = sec + nano / 1e9

    lidar_points = read_points(str(lidar_path), dim=4)  # N,4
    N = lidar_points.shape[0]
    ts_column = np.full((N, 1), ts, dtype=np.float32)
    lidar_points_with_ts = np.hstack([lidar_points, ts_column])

    theta = np.deg2rad(rot_pc)
    R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]], dtype=np.float32)
    lidar_points_with_ts[:, :3] = lidar_points_with_ts[:, :3] @ R_z.T

    output_root = data_root / 'lidar_pc_with_ts'
    output_path = output_root / pc_rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lidar_points_with_ts.astype(np.float32).tofile(str(output_path.with_suffix('.bin')))
    return ('ok', str(output_path))


def rotate_point_cloud(data_root, rot_pc, data_prefix, sensor_type, workers):
    data_root = Path(data_root)
    json_path = data_root / data_prefix / 'HSQJN_3.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        frames = json.load(f)

    tasks = [(str(data_root), data_prefix, sensor_type, rot_pc, frame) for frame in frames]

    ok_cnt = 0;
    skip_cnt = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_rotate_one_frame, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Rotating {data_prefix} all pc frames"):
            status, path = fut.result()
            if status == 'ok':
                ok_cnt += 1
            else:
                skip_cnt += 1
    print(f"[rotate] done: ok={ok_cnt}, skipped={skip_cnt}")


def split_dataset_divisible(dataset_dict, ratio, m=16, seed: int = 42, drop_from='val'):
    """
    将数据随机划分为 train/val，要求两者长度都能被 m 整除。
    如总数 N 不是 m 的倍数，会丢弃 N % m 个样本（默认从 val 部分丢）。
    返回: train_keys, val_keys, dropped_keys
    """
    # 1) 取稳定 token 列表（与 sensor 无关）
    tokens = sorted({v['token'] for v in dataset_dict.values()})
    rnd = random.Random(seed)
    rnd.shuffle(tokens)

    N = len(tokens)
    remainder_total = N % m
    N_eff = N - remainder_total  # 有效样本数，能被 m 整除

    # 先按比例在有效样本上取整到 m 的倍数
    train_len_approx = int(round(N_eff * ratio))
    train_len = (train_len_approx // m) * m  # 向下取到 m 的倍数
    # 确保边界
    train_len = max(0, min(train_len, N_eff))
    val_len = N_eff - train_len  # 自然也是 m 的倍数

    # 前 N_eff 用于 train/val，余下 remainder_total 个直接丢弃
    kept_tokens = tokens[:N_eff]
    dropped_tail = tokens[N_eff:]

    train_tokens = kept_tokens[:train_len]
    val_tokens = kept_tokens[train_len:]

    # 断言
    assert len(train_tokens) % m == 0
    assert len(val_tokens) % m == 0

    train_token_set = set(train_tokens)
    val_token_set = set(val_tokens)

    # 用 token 把原始 key（pc_rel_path）分配到 train/val
    train_keys, val_keys = [], []
    for k, v in dataset_dict.items():
        tok = v['token']
        if tok in train_token_set:
            train_keys.append(k)
        elif tok in val_token_set:
            val_keys.append(k)

    return train_keys, val_keys, dropped_tail


def project_lidar_to_image(img_path, lidar_points, lidar2image, lidar2camera=None, K3=None,
                           max_points=200000):
    """
    在图像上叠加可见的 LiDAR 点。
    - lidar_points: (N, >=3) xyz...[m]
    - lidar2image: (3,4) = K3 @ [R|t]
    - lidar2camera: (4,4) （可选，用于严格用 Z_cam>0 做可见性）
    - K3: (3,3) （可选，用于回投影自检）
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    pts = lidar_points[:, :3].astype(np.float32)
    N = pts.shape[0]
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        pts = pts[idx]

    # 齐次
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.hstack([pts, ones])  # (n,4)

    # 计算相机深度 Zc（可选但推荐用来剔除背面点）
    if lidar2camera is not None:
        Xc = (lidar2camera @ pts_h.T).T  # (n,4)
        Zc = Xc[:, 2]
    else:
        # 近似用 w' 作为深度（仅当 K 第三行接近 [0,0,1] 时才严格成立）
        uvw = (lidar2image @ pts_h.T).T  # (n,3)
        Zc = uvw[:, 2]

    # 投影
    uvw = (lidar2image @ pts_h.T).T  # (n,3)
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]

    # 视野裁剪
    mask = (Zc > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_i = u[mask].astype(np.int32)
    v_i = v[mask].astype(np.int32)
    Zc_vis = Zc[mask]

    # 颜色按深度上色（近=亮，远=暗）
    Zc_norm = np.clip((Zc_vis - np.percentile(Zc_vis, 5)) /
                      (np.percentile(Zc_vis, 95) - np.percentile(Zc_vis, 5) + 1e-6), 0, 1)
    for ui, vi, zn in zip(u_i, v_i, Zc_norm):
        c = int(255 * (1 - zn))
        cv2.circle(img, (ui, vi), 3, (0, c, 255 - c), -1)

    # 可选：回投影自检（把像素+Zc恢复成相机坐标，再用 camera2lidar 回去）
    reproj_rmse = None
    if (K3 is not None) and (lidar2camera is not None):
        fx, fy = K3[0, 0], K3[1, 1]
        cx, cy = K3[0, 2], K3[1, 2]
        u_f = u[mask];
        v_f = v[mask]
        Z_f = Zc[mask]
        x_n = (u_f - cx) / fx
        y_n = (v_f - cy) / fy
        Xc_rec = np.stack([x_n * Z_f, y_n * Z_f, Z_f, np.ones_like(Z_f)], axis=1)  # (m,4)

        # camera2lidar = inv(lidar2camera)
        cam2lidar = np.linalg.inv(lidar2camera).astype(np.float32)
        Xl_rec = (cam2lidar @ Xc_rec.T).T[:, :3]
        Xl_gt = pts[mask, :3]
        reproj_rmse = float(np.sqrt(np.mean(np.sum((Xl_rec - Xl_gt) ** 2, axis=1))))

    return img, {'num_projected': int(mask.sum()), 'rmse_backproj_m': reproj_rmse}


def make_camera_frustum_in_lidar(K3, img_size, camera2lidar, near=0.5, far=30.0):
    """
    生成相机视锥（LineSet），坐标在 LiDAR 系。
    K3: 3x3
    img_size: (w, h)
    camera2lidar: 4x4
    """
    w, h = img_size
    fx, fy = K3[0, 0], K3[1, 1]
    cx, cy = K3[0, 2], K3[1, 2]

    def corners_at(depth):
        # 图像四角像素 -> 归一化光线 -> 乘以 depth 得到相机坐标
        uv = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        x = (uv[:, 0] - cx) / fx
        y = (uv[:, 1] - cy) / fy
        Xc = np.stack([x * depth, y * depth, np.full_like(x, depth), np.ones_like(x)], axis=1)  # (4,4)
        return Xc

    near_c = corners_at(near)
    far_c = corners_at(far)

    # 相机原点
    C = np.array([[0, 0, 0, 1]], dtype=np.float32)

    # 变到 LiDAR
    T = camera2lidar
    near_l = (T @ near_c.T).T[:, :3]
    far_l = (T @ far_c.T).T[:, :3]
    C_l = (T @ C.T).T[:, :3][0]

    # 顶点集合：相机中心 + 4个近面 + 4个远面
    points = np.vstack([C_l[None], near_l, far_l])  # (1+4+4, 3)
    # 线段索引（LineSet 的 pairs）
    idxC = 0
    idxNear = [1, 2, 3, 4]
    idxFar = [5, 6, 7, 8]
    lines = []

    # 从相机中心到近面四角
    for i in idxNear:
        lines.append([idxC, i])

    # 近面四边
    lines += [[1, 2], [2, 3], [3, 4], [4, 1]]
    # 远面四边
    lines += [[5, 6], [6, 7], [7, 8], [8, 5]]
    # 近远对应边
    for i in range(4):
        lines.append([idxNear[i], idxFar[i]])

    # LineSet
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))

    # 给点/线上个色
    colors = np.tile(np.array([[1.0, 0.3, 0.0]]), (len(lines), 1))  # 橙色
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_axes_at_pose(camera2lidar, scale=1.0):
    # 在 LiDAR 系里画出相机坐标轴
    R = camera2lidar[:3, :3]
    t = camera2lidar[:3, 3]
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    # 把世界原点坐标系放到相机位姿：先绕原点旋转R，再平移t
    frame.rotate(R, center=(0, 0, 0))
    frame.translate(t)
    return frame


def visualize_in_pointcloud(lidar_points, K3, img_size, camera2lidar):
    # 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
    # 可选：按高度着色
    z = lidar_points[:, 2]
    z_min, z_max = np.percentile(z, 2), np.percentile(z, 98)
    z_norm = (np.clip(z, z_min, z_max) - z_min) / (z_max - z_min + 1e-6)
    colors = np.stack([z_norm, 1 - z_norm, 0.5 * np.ones_like(z_norm)], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    frustum = make_camera_frustum_in_lidar(K3, img_size, camera2lidar, near=0.5, far=30.0)
    axes = make_axes_at_pose(camera2lidar, scale=0.5)

    o3d.visualization.draw_geometries([pcd, frustum, axes])


def check_img_to_lidar(pc_dataset, pc_key, cam_key='CAM_FRONT'):
    entry = pc_dataset[pc_key]
    cam = entry['cams'][cam_key]
    K3 = cam['camera_intrinsics'][:3, :3] if cam['camera_intrinsics'].shape[0] == 4 else cam['camera_intrinsics']
    camera2lidar = cam['camera2lidar']
    img_path = cam['image_paths']
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    lidar_points = read_points(str(entry['lidar_path']), dim=5)
    visualize_in_pointcloud(lidar_points, K3, (w, h), camera2lidar)


def visualize_gt_boxes_2d(entry, draw_all=False, thickness=2):
    """
    entry: pc_dataset[pc_key]，需要包含
      - entry['cams'][cam_key]['image_paths']
      - entry['gt_boxes_2d'] (N,5)  [x1,y1,x2,y2,cam_index]  (过滤前长度)
      - entry['empty_mask'] (N,)  用于对齐过滤后3D（可选，但建议有）
      - entry['cam_keys'] 生成2D框时保存的 cam_key 顺序（强烈建议）
    out_dir: 保存输出图片的目录
    draw_all: True 时也会画 entry['all_2d_boxes']
    """

    if "gt_boxes_2d" not in entry:
        raise KeyError("entry 中没有 gt_boxes_2d，请先在 dataset 构建时生成 2D 框。")

    cam_keys = entry.get("cam_keys", None)
    if cam_keys is None:
        # fallback：用 dict keys（可能不稳定）
        cam_keys = list(entry["cams"].keys())

    # 读取每个相机图片
    imgs = {}
    for ck in cam_keys:
        img_path = entry["cams"][ck]["image_paths"]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] failed to read image: {img_path}")
            continue
        imgs[ck] = img

    # 只画过滤后3D对应的2D框（与 create_groundtruth_database 对齐逻辑一致）
    boxes2d = entry["gt_boxes_2d"]
    if "empty_mask" in entry:
        boxes2d = boxes2d[entry["empty_mask"]]

    # 逐框画到对应相机
    for i in range(boxes2d.shape[0]):
        x1, y1, x2, y2, cam_idx = boxes2d[i].tolist()
        cam_idx = int(cam_idx)

        if cam_idx < 0 or cam_idx >= len(cam_keys):
            # dummy 或异常 index
            continue

        ck = cam_keys[cam_idx]
        if ck not in imgs:
            continue

        img = imgs[ck]
        h, w = img.shape[:2]
        x1i = int(np.clip(x1, 0, w - 1))
        y1i = int(np.clip(y1, 0, h - 1))
        x2i = int(np.clip(x2, 0, w - 1))
        y2i = int(np.clip(y2, 0, h - 1))

        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0, 255, 0), thickness)
        cv2.putText(img, f"id={i}", (x1i, max(y1i - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 可选：画 all_2d_boxes（所有相机上的投影框）
    if draw_all and ("all_2d_boxes" in entry):
        all_boxes = entry["all_2d_boxes"]
        for j in range(all_boxes.shape[0]):
            x1, y1, x2, y2, cam_idx = all_boxes[j].tolist()
            cam_idx = int(cam_idx)
            if cam_idx < 0 or cam_idx >= len(cam_keys):
                continue
            ck = cam_keys[cam_idx]
            if ck not in imgs:
                continue
            img = imgs[ck]
            h, w = img.shape[:2]
            x1i = int(np.clip(x1, 0, w - 1))
            y1i = int(np.clip(y1, 0, h - 1))
            x2i = int(np.clip(x2, 0, w - 1))
            y2i = int(np.clip(y2, 0, h - 1))
            cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (255, 0, 0), 1)

    for ck in cam_keys:
        if ck in imgs:
            cv2.imshow(f"2D GT in {ck}", imgs[ck])
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()


def check_lidar_to_img(pc_dataset, pc_key, cam_key='CAM_FRONT'):
    entry = pc_dataset[pc_key]
    cam = entry['cams'][cam_key]
    img_path = cam['image_paths']
    lidar2image = cam['lidar2image'][:3, :] if cam['lidar2image'].shape[0] == 4 else cam['lidar2image']
    lidar2camera = cam['lidar2camera']
    # 你的 camera_intrinsics 可能是 4x4，这里取 3x3
    K3 = cam['camera_intrinsics'][:3, :3] if cam['camera_intrinsics'].shape[0] == 4 else cam['camera_intrinsics']

    # 读取点云（按你的读取函数）
    lidar_path = entry['lidar_path']
    lidar_points = read_points(str(lidar_path), dim=5)  # 你已有的函数

    img, stats = project_lidar_to_image(img_path, lidar_points, lidar2image, lidar2camera, K3)
    print(f"[overlay] projected={stats['num_projected']}, back-proj RMSE={stats['rmse_backproj_m']:.3f} m")
    cv2.imshow("3d LiDAR pc -> Image (colored by intensity)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def _project_points_uv(lidar_xyz, lidar2image_4x4):
    """lidar_xyz: (N,3) -> uv (N,2), depth (N,)"""
    N = lidar_xyz.shape[0]
    pts_h = np.hstack([lidar_xyz.astype(np.float32), np.ones((N,1), np.float32)])  # (N,4)
    proj = (lidar2image_4x4 @ pts_h.T)[:3, :].T  # (N,3)
    z = proj[:, 2]
    u = proj[:, 0] / (z + 1e-6)
    v = proj[:, 1] / (z + 1e-6)
    return np.stack([u, v], axis=1), z

def _draw_box3d_lines(img, uv8, color=(0,255,0), thickness=2):
    for a,b in _EDGES:
        p1 = tuple(np.round(uv8[a]).astype(int).tolist())
        p2 = tuple(np.round(uv8[b]).astype(int).tolist())
        cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

def vis_object_points_and_box3d_on_image(
    info,                   # sunlakes_infos_train.pkl 里的一条
    obj_i: int,             # 第 i 个 gt box
    gt_points_local: np.ndarray,  # 你那段代码里已经减中心后的 gt_points (M,5)
    cam_key: str = None,    # 可指定相机；None 则用 2D cam_index 自动选
    max_points: int = 5000
):
    """
    复用 project_lidar_to_image 画点；再把 box3d 角点投影画线框。
    """

    gt_boxes = info['gt_boxes']
    gt_names = info.get('gt_names', None)

    if obj_i < 0 or obj_i >= gt_boxes.shape[0]:
        return

    box = gt_boxes[obj_i].astype(np.float32)  # (>=7)

    # 选相机：优先按 2D GT 的 cam_index
    cam_keys = info.get("cam_keys", list(info["cams"].keys()))
    if cam_key is None:
        cam_key = cam_keys[0]
        if ('gt_boxes_2d' in info) and ('empty_mask' in info):
            gt2d_kept = info['gt_boxes_2d'][info['empty_mask']]
            if obj_i < gt2d_kept.shape[0]:
                cam_idx = int(gt2d_kept[obj_i, 4])
                if 0 <= cam_idx < len(cam_keys):
                    cam_key = cam_keys[cam_idx]

    cam = info["cams"][cam_key]
    img_path = cam["image_paths"]
    lidar2image = cam["lidar2image"]
    lidar2camera = cam.get("lidar2camera", None)
    K3 = cam["camera_intrinsics"][:3, :3] if cam["camera_intrinsics"].shape[0] == 4 else cam["camera_intrinsics"]

    # ✅ 把局部点云加回全局中心，才能正确投影
    pts = gt_points_local.copy()
    pts[:, :3] += box[:3]

    # 下采样，避免太密
    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]

    # 用你现成的 project_lidar_to_image 画“该物体点云”
    # 注意：project_lidar_to_image 读图并返回 img，我们不在里面 show，改成保存
    img, stats = project_lidar_to_image(
        img_path=img_path,
        lidar_points=pts,                       # 只投影该 bbox 内点
        lidar2image=lidar2image[:3, :] if lidar2image.shape[0] == 4 else lidar2image,
        lidar2camera=lidar2camera,
        K3=K3,
        max_points=max_points
    )

    # 再画该物体的 3D box 线框
    box7 = box[:7][None, :].astype(np.float32)
    corners = corners_for_projection_from_gtboxes(box7, yaw_flip=False)[0]   # (3,8)
    corners_xyz = corners.T  # (8,3)
    uv8, z8 = _project_points_uv(corners_xyz, lidar2image.astype(np.float32))
    H, W = img.shape[:2]

    # 角点全在相机后面就不画
    if not np.all(z8 <= 0):
        uv8[:,0] = np.clip(uv8[:,0], 0, W-1)
        uv8[:,1] = np.clip(uv8[:,1], 0, H-1)
        _draw_box3d_lines(img, uv8, color=(0,255,0), thickness=2)

    # 标注信息
    name = str(gt_names[obj_i]) if gt_names is not None else "obj"
    cv2.putText(img, f"{name} idx={obj_i} {cam_key} pts={stats['num_projected']}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("3d GT -> image", img)
    key = cv2.waitKey(0) & 0xFF   # 按任意键下一张
    cv2.destroyAllWindows()
