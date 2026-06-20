# -*- coding: utf-8 -*-
"""Export Sunlakes roadside multi-LiDAR data to OpenCOOD format.

This exporter treats os128b as the ego agent and exports the remaining
fixed roadside LiDARs as cooperative agents. By default, all point clouds
are assumed to be pre-registered in the same os128b/world frame, so every
agent uses an identity lidar_pose. If you have static per-sensor poses,
provide them with --sensor_pose_path.
"""

import argparse
import json
import pickle
import random
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CAM_TYPES = {
    "axis1_label": "CAM_FRONT",
    "axis3_label": "CAM_FRONT_LEFT",
    "axis4_label": "CAM_FRONT_RIGHT",
    "axis2_label": "CAM_BACK",
}

BOX_SOURCE_LABEL = "128_128b_label"
SENSOR_LABEL_MAP = {
    "os128": "ouster128_label",
    "os128b": "ouster128b_label",
    "os64": "ouster64_label",
    "os64b": "ouster64b_label",
    "rs16": "rslidar16_label",
    "rs16b": "rslidar16b_label",
}
KNOWN_LABEL_SEGMENTS = {
    BOX_SOURCE_LABEL,
    "ouster128_label",
    "ouster128b_label",
    "ouster64_label",
    "ouster64b_label",
    "rslidar16_label",
    "rslidar16b_label",
    "128_128b_label",
    "64_64b_label",
    "16_16b_label",
    "64_64b_downsample",
    "16_16b_downsample",
}
DEFAULT_AGENT_ORDER = ["os128b", "os128", "os64", "os64b", "rs16", "rs16b"]
DEFAULT_COOPERATIVE_AGENTS = []
NAME_REMAP = {
    "van": "barrier",
    "construction vehicle": "construction_vehicle",
    "golf cart": "traffic_cone",
}


def replace_label_segment(pc_rel_path, target_segment):
    parts = list(Path(pc_rel_path).parts)
    for i, seg in enumerate(parts):
        if seg in KNOWN_LABEL_SEGMENTS:
            parts[i] = target_segment
            return str(Path(*parts))
    return str(Path(pc_rel_path))


def resolve_sensor_pc_rel_path(json_pc_rel_path, sensor_type):
    if sensor_type not in SENSOR_LABEL_MAP:
        raise ValueError(f"Unsupported sensor_type: {sensor_type}")
    return replace_label_segment(json_pc_rel_path, SENSOR_LABEL_MAP[sensor_type])


def resolve_box_pc_rel_path(json_pc_rel_path):
    return replace_label_segment(json_pc_rel_path, BOX_SOURCE_LABEL)


def read_points(file_path, dim=4):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)


def pc_id_to_ts(pc_id):
    sec, nano = map(int, pc_id.split("-"))
    return sec + nano / 1e9


def rotate_xyz(xyz, theta):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return xyz @ r.T


def load_sensor_pose_map(sensor_pose_path, agents, ego_sensor):
    default_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pose_map = {sensor: list(default_pose) for sensor in agents}
    pose_map[ego_sensor] = list(default_pose)

    if sensor_pose_path is None:
        print("[warn] sensor_pose_path not provided; all agents use identity lidar_pose. "
              "This assumes all exported point clouds are already registered in os128b frame.")
        return pose_map

    path = Path(sensor_pose_path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    for sensor in agents:
        value = raw.get(sensor)
        if value is None:
            continue
        if isinstance(value, dict):
            value = value.get("lidar_pose", value.get("pose"))
        if not isinstance(value, (list, tuple)) or len(value) != 6:
            raise ValueError(f"Pose for {sensor} must be a list of 6 floats, got {value}")
        pose_map[sensor] = [float(v) for v in value]

    return pose_map


def normalize_intensity(intensity):
    intensity = np.nan_to_num(intensity.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if intensity.size == 0:
        return intensity
    if intensity.min() >= 0.0 and intensity.max() <= 1.0:
        return intensity
    lo = float(intensity.min())
    hi = float(intensity.max())
    if hi - lo < 1e-6:
        return np.zeros_like(intensity)
    return np.clip((intensity - lo) / (hi - lo), 0.0, 1.0)


def write_pcd(save_path, points):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    intensity = normalize_intensity(points[:, 3])
    colors = np.repeat(intensity[:, None], 3, axis=1).astype(np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    ok = o3d.io.write_point_cloud(str(save_path), pcd, write_ascii=False, compressed=False)
    if not ok:
        raise RuntimeError(f"Failed to write pcd: {save_path}")


def sanitize_name(value):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)


def load_points_in_boxes_cpu():
    try:
        from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
        return points_in_boxes_cpu
    except Exception as exc:
        print(f"[warn] points_in_boxes_cpu unavailable, falling back to numpy box counting: {exc}")
        return None


def count_points_in_boxes_numpy(points_xyz, boxes7):
    counts = np.zeros((boxes7.shape[0],), dtype=np.int32)
    for idx, box in enumerate(boxes7):
        cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in box]
        shifted = points_xyz - np.array([cx, cy, cz], dtype=np.float32)
        c, s = np.cos(yaw), np.sin(yaw)
        local_x = shifted[:, 0] * c + shifted[:, 1] * s
        local_y = -shifted[:, 0] * s + shifted[:, 1] * c
        local_z = shifted[:, 2]
        mask = (
            (np.abs(local_x) <= dx / 2.0)
            & (np.abs(local_y) <= dy / 2.0)
            & (np.abs(local_z) <= dz / 2.0)
        )
        counts[idx] = int(mask.sum())
    return counts


def count_points_in_boxes(points_xyz, boxes7, points_in_boxes_cpu):
    if boxes7.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    if points_in_boxes_cpu is None:
        return count_points_in_boxes_numpy(points_xyz, boxes7)
    point_indices = points_in_boxes_cpu(points_xyz.astype(np.float32), boxes7.astype(np.float32))
    return point_indices.sum(axis=1).astype(np.int32)


def frame_to_boxes(frame, resource_root, rot_pc_deg, points_in_boxes_cpu):
    json_pc_rel_path = "/".join(frame["info"].split("/")[-3:])
    box_pc_rel_path = resolve_box_pc_rel_path(json_pc_rel_path)
    box_lidar_path = resource_root / box_pc_rel_path
    if not box_lidar_path.exists():
        return None

    box3d_labels = [label for label in frame.get("labels", []) if label.get("drawType") == "box3d"]
    if not box3d_labels:
        return None

    box_points = read_points(str(box_lidar_path), dim=4)
    theta = np.deg2rad(rot_pc_deg)
    box_points[:, :3] = rotate_xyz(box_points[:, :3], theta)

    locs = np.asarray([label["points"][:3] for label in box3d_labels], dtype=np.float32)
    dims = np.asarray([label["points"][6:] for label in box3d_labels], dtype=np.float32)
    rots = np.asarray([label["points"][5] for label in box3d_labels], dtype=np.float32)
    names = np.asarray([NAME_REMAP.get(label["label"], label["label"]) for label in box3d_labels], dtype=object)
    box_ids = np.asarray([int(label.get("id", idx)) for idx, label in enumerate(box3d_labels)], dtype=np.int64)

    locs = rotate_xyz(locs, theta)
    rots = rots + theta

    boxes7 = np.concatenate([locs, dims, rots[:, None]], axis=1)
    num_lidar_pts = count_points_in_boxes(box_points[:, :3], boxes7, points_in_boxes_cpu)
    mask = num_lidar_pts > 0
    if not np.any(mask):
        return None

    token = Path(json_pc_rel_path).stem
    return {
        "json_pc_rel_path": json_pc_rel_path,
        "token": token,
        "timestamp": token,
        "timestamp_sec": float(pc_id_to_ts(token)),
        "frame_index": int(frame.get("frameIndex", 0)),
        "scene_id": str(frame.get("batchId", "")),
        "box_ids": box_ids,
        "locs": locs,
        "dims": dims,
        "rots": rots,
        "names": names,
        "num_lidar_pts_full": num_lidar_pts.astype(np.int32),
        "keep_mask": mask.astype(bool),
    }


def finalize_frame_records(parsed_frames):
    prev_state = {}
    first_frame_pending = {}
    first_frame_index = None
    finalized = []

    for frame in parsed_frames:
        locs = frame["locs"]
        dims = frame["dims"]
        rots = frame["rots"]
        names = frame["names"]
        box_ids = frame["box_ids"]
        mask = frame["keep_mask"]
        num_lidar_pts = frame["num_lidar_pts_full"]
        ts = frame["timestamp_sec"]
        scene_id = frame["scene_id"]

        vxvy_full = np.zeros((len(box_ids), 2), dtype=np.float32)
        for j, oid in enumerate(box_ids):
            xy = locs[j, :2].astype(np.float32)
            key = (scene_id, int(oid))
            if key in prev_state:
                prev_xy, prev_ts = prev_state[key]
                dt = float(ts - prev_ts)
                if 1e-6 < dt < 2.0:
                    vxvy_full[j, 0] = (xy[0] - prev_xy[0]) / dt
                    vxvy_full[j, 1] = (xy[1] - prev_xy[1]) / dt
            prev_state[key] = (xy, ts)

        gt_boxes_full = np.concatenate([locs, dims, rots[:, None], vxvy_full], axis=1)
        kept_ids = box_ids[mask]
        gt_boxes = gt_boxes_full[mask].astype(np.float32)
        velocitys = gt_boxes[:, 7:9].astype(np.float32)
        kept_names = names[mask]
        kept_num_lidar_pts = num_lidar_pts[mask].astype(np.int32)
        id_to_kept_idx = {int(oid): int(i) for i, oid in enumerate(kept_ids)}

        record = {
            "json_pc_rel_path": frame["json_pc_rel_path"],
            "token": frame["token"],
            "timestamp": frame["timestamp"],
            "timestamp_sec": ts,
            "frame_index": frame["frame_index"],
            "scene_id": scene_id,
            "vehicles": OrderedDict(),
            "gt_boxes": gt_boxes,
            "gt_boxes_velocity": velocitys,
            "gt_names": kept_names,
            "num_lidar_pts": kept_num_lidar_pts,
            "tracking_id": [f"{scene_id}_{int(oid)}" for oid in kept_ids],
        }

        for idx, oid in enumerate(kept_ids.tolist()):
            dx, dy, dz = [float(v) for v in dims[mask][idx]]
            x, y, z = [float(v) for v in locs[mask][idx]]
            yaw = float(rots[mask][idx])
            record["vehicles"][str(oid)] = {
                "angle": [0.0, yaw, 0.0],
                "center": [0.0, 0.0, 0.0],
                "extent": [dx / 2.0, dy / 2.0, dz / 2.0],
                "location": [x, y, z],
                "type": str(kept_names[idx]),
                "num_points": int(kept_num_lidar_pts[idx]),
            }

        fidx = frame["frame_index"]
        if first_frame_index is None or fidx < first_frame_index:
            first_frame_index = fidx
            first_frame_pending = {"record": record, "id2idx": dict(id_to_kept_idx), "idx": fidx}
        elif first_frame_pending and fidx == first_frame_pending["idx"] + 1:
            prev_record = first_frame_pending["record"]
            prev_map = first_frame_pending["id2idx"]
            for oid, cur_idx in id_to_kept_idx.items():
                if oid in prev_map:
                    prev_idx = prev_map[oid]
                    v2 = record["gt_boxes_velocity"][cur_idx].astype(np.float32)
                    prev_record["gt_boxes_velocity"][prev_idx] = v2
                    prev_record["gt_boxes"][prev_idx, 7:9] = v2
            first_frame_pending = None

        finalized.append(record)

    return finalized


def build_openpcdet_infos(selected_names, scenarios):
    infos = []
    for scenario_name in sorted(selected_names):
        for frame in scenarios[scenario_name]["frames"]:
            infos.append({
                "lidar_path": str(frame["json_pc_rel_path"]),
                "token": frame["token"],
                "scene_id": frame["scene_id"],
                "timestamp": frame["timestamp_sec"],
                "tracking_id": list(frame["tracking_id"]),
                "gt_boxes": frame["gt_boxes"].astype(np.float32),
                "gt_boxes_velocity": frame["gt_boxes_velocity"].astype(np.float32),
                "gt_names": np.asarray(frame["gt_names"], dtype=object),
                "num_lidar_pts": np.asarray(frame["num_lidar_pts"], dtype=np.int32),
            })
    return infos


def dump_openpcdet_infos(output_root, train_infos, val_infos):
    eval_root = output_root / "openpcdet_eval" / "v1.0-trainval"
    eval_root.mkdir(parents=True, exist_ok=True)
    with open(eval_root / "sunlakes_infos_train.pkl", "wb") as f:
        pickle.dump(train_infos, f)
    with open(eval_root / "sunlakes_infos_val.pkl", "wb") as f:
        pickle.dump(val_infos, f)
    return eval_root


def gather_scenarios(data_root, prefixes, rot_pc_deg, max_frames_per_scenario):
    points_in_boxes_cpu = load_points_in_boxes_cpu()
    scenarios = OrderedDict()

    for data_prefix in tqdm(prefixes, desc="Scanning prefixes"):
        prefix_root = data_root / data_prefix
        resource_root = prefix_root / "resource"
        json_path = prefix_root / "HSQJN_3.json"
        with open(json_path, "r", encoding="utf-8") as f:
            frames = json.load(f)

        bucket = defaultdict(list)
        for frame in frames:
            bucket[frame["batchId"]].append(frame)

        for batch_id, batch_frames in tqdm(sorted(bucket.items()), desc=f"Parsing {data_prefix} scenarios", leave=False):
            ordered_frames = sorted(
                batch_frames,
                key=lambda item: (int(item.get("frameIndex", 0)), "/".join(item["info"].split("/")[-3:])),
            )
            if max_frames_per_scenario > 0:
                ordered_frames = ordered_frames[:max_frames_per_scenario]

            scenario_name = sanitize_name(f"{data_prefix}__{batch_id}")
            parsed_frames = []
            for frame in tqdm(ordered_frames, desc=f"Frames {scenario_name}", leave=False):
                parsed = frame_to_boxes(frame, resource_root, rot_pc_deg, points_in_boxes_cpu)
                if parsed is not None:
                    parsed_frames.append(parsed)

            if parsed_frames:
                scenarios[scenario_name] = {
                    "data_prefix": data_prefix,
                    "resource_root": resource_root,
                    "frames": finalize_frame_records(parsed_frames),
                }

    return scenarios


def split_scenarios(scenario_names, train_ratio, seed):
    names = list(scenario_names)
    rnd = random.Random(seed)
    rnd.shuffle(names)
    train_count = int(round(len(names) * train_ratio))
    train_count = min(max(train_count, 1 if len(names) > 1 else len(names)), len(names))
    if train_count == len(names) and len(names) > 1:
        train_count -= 1
    train_set = set(names[:train_count])
    val_set = set(names[train_count:])
    if not val_set and train_set:
        moved = next(iter(train_set))
        train_set.remove(moved)
        val_set.add(moved)
    return train_set, val_set


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dump_yaml(save_path, lidar_pose, vehicles):
    payload = {
        "lidar_pose": [float(v) for v in lidar_pose],
        "ego_speed": 0.0,
        "vehicles": vehicles,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_builtin(payload), f, sort_keys=False)


def export_split(output_root, split_name, scenarios, selected_names, agents, agent_id_map, pose_map, rot_pc_deg):
    split_root = output_root / split_name
    theta = np.deg2rad(rot_pc_deg)
    exported_frames = 0
    total_frames = sum(len(scenarios[name]["frames"]) for name in selected_names)

    for scenario_name in tqdm(sorted(selected_names), desc=f"Preparing {split_name} scenarios"):
        scenario_root = split_root / scenario_name
        for agent in agents:
            (scenario_root / str(agent_id_map[agent])).mkdir(parents=True, exist_ok=True)

    with tqdm(total=total_frames, desc=f"Exporting {split_name} frames") as pbar:
        for scenario_name in sorted(selected_names):
            scenario = scenarios[scenario_name]
            scenario_root = split_root / scenario_name
            resource_root = scenario["resource_root"]

            for frame in scenario["frames"]:
                json_pc_rel_path = frame["json_pc_rel_path"]
                timestamp = frame["timestamp"]
                vehicles = frame["vehicles"]

                for agent in agents:
                    cav_dir = scenario_root / str(agent_id_map[agent])
                    source_rel_path = resolve_sensor_pc_rel_path(json_pc_rel_path, agent)
                    source_path = resource_root / source_rel_path
                    if not source_path.exists():
                        raise FileNotFoundError(f"Missing source lidar for {agent}: {source_path}")

                    points = read_points(str(source_path), dim=4)
                    points[:, :3] = rotate_xyz(points[:, :3], theta)
                    write_pcd(cav_dir / f"{timestamp}.pcd", points)
                    dump_yaml(cav_dir / f"{timestamp}.yaml", pose_map[agent], vehicles)

                exported_frames += 1
                pbar.update(1)

    return exported_frames


def write_manifest(output_root, agents, agent_id_map, pose_map, train_names, val_names, args, opencood_eval_root):
    manifest = {
        "dataset": "sunlakes_opencood",
        "ego_sensor": args.ego_sensor,
        "agents": agents,
        "agent_id_map": agent_id_map,
        "sensor_pose": pose_map,
        "box_source_label": BOX_SOURCE_LABEL,
        "train_scenarios": sorted(train_names),
        "validate_scenarios": sorted(val_names),
        "data_prefix": args.data_prefix,
        "rot_pc": float(args.rot_pc),
        "aligned_openpcdet_eval_root": str(opencood_eval_root),
    }
    with open(output_root / "sunlakes_opencood_manifest.json", "w", encoding="utf-8") as f:
        json.dump(to_builtin(manifest), f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Export Sunlakes data to OpenCOOD format")
    parser.add_argument("--data_root", type=str, default="/home/suolab/OpenPCDet/data/sunlakes", help="raw Sunlakes root directory")
    parser.add_argument("--output_root", type=str, default="/home/suolab/OpenCOOD/data/sunlakes_opencood_os128b_os128_2026-01-04_18_02_01-export", help="OpenCOOD-format output root")
    parser.add_argument("--data_prefix", nargs="+", default=["2026-01-04_18_02_01-export"], help="one or more Sunlakes subpaths")
    parser.add_argument("--rot_pc", type=float, default=-142.5, help="rotation angle applied to raw point clouds")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="scenario-level train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed for train/validate split")
    parser.add_argument("--max_frames_per_scenario", type=int, default=0,
                        help="optional cap for each scenario, 0 means all frames")
    parser.add_argument("--ego_sensor", type=str, default="os128b", choices=DEFAULT_AGENT_ORDER,
                        help="sensor treated as ego agent in OpenCOOD")
    parser.add_argument("--cooperative_agents", nargs="+", default=DEFAULT_COOPERATIVE_AGENTS,
                        help="cooperative sensors to export alongside ego")
    parser.add_argument("--agents", nargs="+", default=None,
                        help="explicit full sensor list to export; overrides ego/cooperative settings")
    parser.add_argument("--sensor_pose_path", type=str, default=None,
                        help="optional JSON/YAML mapping sensor -> [x,y,z,roll,yaw,pitch]")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    requested_agents = args.agents if args.agents is not None else [args.ego_sensor] + list(args.cooperative_agents)
    agents = []
    for sensor in requested_agents:
        if sensor not in SENSOR_LABEL_MAP:
            raise ValueError(f"Unsupported sensor: {sensor}")
        if sensor not in agents:
            agents.append(sensor)
    if args.ego_sensor not in agents:
        agents.insert(0, args.ego_sensor)
    else:
        agents.remove(args.ego_sensor)
        agents.insert(0, args.ego_sensor)

    pose_map = load_sensor_pose_map(args.sensor_pose_path, agents, args.ego_sensor)
    agent_id_map = {sensor: idx for idx, sensor in enumerate(agents)}

    scenarios = gather_scenarios(
        data_root=data_root,
        prefixes=args.data_prefix,
        rot_pc_deg=args.rot_pc,
        max_frames_per_scenario=args.max_frames_per_scenario,
    )
    if not scenarios:
        raise RuntimeError("No valid scenarios found for export")

    train_names, val_names = split_scenarios(scenarios.keys(), args.train_ratio, args.seed)
    train_frames = export_split(output_root, "train", scenarios, train_names, agents, agent_id_map, pose_map, args.rot_pc)
    val_frames = export_split(output_root, "validate", scenarios, val_names, agents, agent_id_map, pose_map, args.rot_pc)
    train_infos = build_openpcdet_infos(train_names, scenarios)
    val_infos = build_openpcdet_infos(val_names, scenarios)
    opencood_eval_root = dump_openpcdet_infos(output_root, train_infos, val_infos)
    write_manifest(output_root, agents, agent_id_map, pose_map, train_names, val_names, args, opencood_eval_root)

    print(f"Export finished: train_scenarios={len(train_names)}, validate_scenarios={len(val_names)}, "
          f"train_frames={train_frames}, validate_frames={val_frames}, opencood_eval_root={opencood_eval_root}")


if __name__ == "__main__":
    main()
