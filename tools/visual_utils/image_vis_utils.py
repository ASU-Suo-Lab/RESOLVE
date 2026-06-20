import re
import numpy as np
from pathlib import Path
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pcdet.utils import box_utils

INTRINSIC_PAT = re.compile(r"Intrinsic:\s*\n([^\n]+)\n([^\n]+)\n([^\n]+)")
DISTORT_PAT = re.compile(r"Distortion:\s*\n\[(.*?)\]", re.S)
R_PAT = re.compile(r"R:\s*(.*?)\s*t:", re.S)
T_PAT = re.compile(r"t:\s*(.*?)\n")

def rotmat_z_deg(rot_pc_deg: float) -> np.ndarray:
    th = np.deg2rad(rot_pc_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)
                     
def rgb01_to_bgr255(rgb01):
    rgb01 = np.array(rgb01, dtype=np.float32)
    rgb01 = np.clip(rgb01, 0.0, 1.0)
    bgr = (rgb01[::-1] * 255.0).astype(np.uint8)  # RGB->BGR
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))
    

def get_extrinsic(calib_path):
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


def get_cam_kd(calib_path):
    content = Path(calib_path).read_text(encoding="utf-8")
    m = INTRINSIC_PAT.search(content)
    if m is None:
        raise ValueError(f"Intrinsic 未找到: {calib_path}")
    K = np.vstack([np.fromstring(row, sep=" ") for row in m.groups()]).astype(np.float32)

    m = DISTORT_PAT.search(content)
    D = np.fromstring(m.group(1), sep=",", dtype=np.float32) if m else np.zeros(5, np.float32)
    return K, D

def project_points_no_image_mask(points_3d: np.ndarray, K: np.ndarray, extrinsic: np.ndarray):
    R_ext = extrinsic[:3, :3]
    t_ext = extrinsic[:3, 3]

    pts_cam = (R_ext @ points_3d.T) + t_ext.reshape(3, 1)   # (3,N)
    z = pts_cam[2, :]

    proj = K @ pts_cam
    u = proj[0, :] / (proj[2, :] + 1e-6)
    v = proj[1, :] / (proj[2, :] + 1e-6)

    uvz = np.stack([u, v, z], axis=1)
    valid_z = z > 0
    return uvz, valid_z

def pcdet_boxes_to_o3d_obbs(boxes_lidar):
    """
    boxes_lidar: (N,7) [x,y,z,dx,dy,dz,heading]
    return: list[o3d.geometry.OrientedBoundingBox]
    """
    boxes = np.asarray(boxes_lidar, dtype=np.float32)
    obbs = []
    for b in boxes:
        center = b[0:3]
        extent = b[3:6]      # dx,dy,dz
        yaw = float(b[6])    # heading about z
        rot = R.from_euler("z", yaw, degrees=False).as_matrix()
        obb = o3d.geometry.OrientedBoundingBox(center=center, R=rot, extent=extent)
        obbs.append(obb)
    return obbs


EDGES = [
    (0, 1), (1, 7), (7, 2), (2, 0),
    (4, 5), (5, 3), (3, 6), (6, 4),
    (0, 3), (1, 6), (2, 5), (4, 7)
]

def draw_obbs_on_image(out_img, obbs, K, extrinsic, color=(0,255,0), thickness=2):
    h, w = out_img.shape[:2]
    rect = (0, 0, w, h)

    for obb in obbs:
        corners = np.asarray(obb.get_box_points(), dtype=np.float32)  # (8,3)
        uvz, valid_z = project_points_no_image_mask(corners, K, extrinsic)
        uv = uvz[:, :2]

        for i, j in EDGES:
            if (not valid_z[i]) or (not valid_z[j]):
                continue

            p1 = (int(round(uv[i, 0])), int(round(uv[i, 1])))
            p2 = (int(round(uv[j, 0])), int(round(uv[j, 1])))

            ok, q1, q2 = cv2.clipLine(rect, p1, p2)
            if ok:
                cv2.line(out_img, q1, q2, color, thickness, cv2.LINE_AA)
