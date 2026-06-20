#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import yaml
import time
import argparse
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# ---- MCTrack core (your repo) ----
from tracker.base_tracker import Base3DTracker
from tracker.bbox import BBox  # adjust path if needed


# -------------------------
# SunLakes classes
# -------------------------
SUNLAKES_TRACKING_NAMES = [
    "barrier",               # van
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",          # golf cart
    "trailer",
    "truck",
]

# If your detector outputs van/golf_cart instead of barrier/traffic_cone, map here.
ALIASES = {
    "van": "barrier",
    "golf_cart": "traffic_cone",
}


# ============================================================
# Part A: Tracking pipeline
# ============================================================
def quat_to_yaw(q):
    """q: [qw,qx,qy,qz] -> yaw(rad) around z."""
    qw, qx, qy, qz = q
    return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def normalize_det_item(d: dict):
    """Make detection dict keys consistent."""
    if "detection_name" not in d and "tracking_name" in d:
        d["detection_name"] = d["tracking_name"]
    if "detection_score" not in d and "tracking_score" in d:
        d["detection_score"] = d["tracking_score"]

    if "velocity" not in d or d["velocity"] is None:
        d["velocity"] = [0.0, 0.0]
    if len(d["velocity"]) >= 2:
        d["velocity"] = d["velocity"][:2]
    else:
        d["velocity"] = [0.0, 0.0]
    return d


def load_frames_from_pkl(data_root: Path, split: str):
    """
    Frames in chronological order, with scene boundary flag.
    Return:
      [{"scene_id":..., "token":..., "timestamp":..., "first":bool}, ...]
    """
    pkl_path = data_root / f"sunlakes_infos_{split}.pkl"
    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)

    by_scene = defaultdict(list)
    for info in infos:
        scene = info.get("scene_id", "scene0")
        ts = float(info["timestamp"])
        token = info["token"]
        by_scene[scene].append((ts, token))

    frames = []
    for scene, seq in by_scene.items():
        seq.sort(key=lambda x: x[0])
        for k, (ts, token) in enumerate(seq):
            frames.append({"scene_id": scene, "token": token, "timestamp": ts, "first": (k == 0)})
    return frames


class SunLakesFrameInfo:
    """
    Minimal frame_info required by Base3DTracker.track_single_frame():
      - frame_info.frame_id
      - frame_info.bboxes (list of MCTrack BBox objects)
      - frame_info.transform_matrix (optional, RV matching)
    """
    __slots__ = ["frame_id", "bboxes", "transform_matrix", "cur_sample_token"]

    def __init__(self, frame_id: int, token: str, bboxes, transform_matrix=None):
        self.frame_id = frame_id
        self.cur_sample_token = token
        self.bboxes = bboxes
        self.transform_matrix = transform_matrix


def make_mctrack_bbox(det: dict, frame_id: int):
    """
    Convert one det dict (nuScenes-like) to MCTrack BBox object.
    det expects:
      translation(x,y,z), size(w,l,h), rotation(qw,qx,qy,qz), velocity(vx,vy),
      detection_name, detection_score
    """
    det = normalize_det_item(det)

    name = det.get("detection_name", None)
    if name is None:
        return None
    name = ALIASES.get(name, name)
    if name not in SUNLAKES_TRACKING_NAMES:
        return None

    score = float(det.get("detection_score", 0.0))

    # nuScenes size is wlh; MCTrack BBox expects lwh
    w, l, h = det["size"]
    lwh = [float(l), float(w), float(h)]

    xyz = [float(det["translation"][0]), float(det["translation"][1]), float(det["translation"][2])]

    rot = [float(x) for x in det["rotation"]]  # [qw,qx,qy,qz]
    yaw = quat_to_yaw(rot)

    vel = det.get("velocity", [0.0, 0.0])
    if vel is None: vel = [0.0, 0.0]
    if len(vel) < 2: vel = [0.0, 0.0]
    vel = [float(vel[0]), float(vel[1])]

    bbox_dict = {
        "category": name,
        "detection_score": score,
        "lwh": lwh,
        "global_xyz": xyz,
        "global_orientation": rot,
        "global_yaw": yaw,
        "global_velocity": vel,
        "global_acceleration": [0.0, 0.0],  # default
        "bbox_image": {},                   # must exist for BBox.__init__()
    }
    return BBox(frame_id=frame_id, bbox=bbox_dict)


def run_tracking(cfg: dict,
                 data_root: Path,
                 split: str,
                 pred_json: Path,
                 work_dir: Path,
                 pred_token_key: str = "results"):
    work_dir.mkdir(parents=True, exist_ok=True)

    # frames order from pkl
    frames = load_frames_from_pkl(data_root, split)

    # detections
    with open(pred_json, "r") as f:
        pred_all = json.load(f)
    pred_by_token = pred_all[pred_token_key]  # token -> list[det]

    out = {"results": {}, "meta": None}

    # per scene reset
    scene_frame_id = defaultdict(int)
    tracker = None

    for fr in tqdm(frames, desc="[MCTrack-SunLakes] Tracking"):
        token = fr["token"]
        scene = fr["scene_id"]

        if fr["first"]:
            tracker = Base3DTracker(cfg=cfg)
            scene_frame_id[scene] = 0

        scene_frame_id[scene] += 1
        fid = scene_frame_id[scene]

        dets = pred_by_token.get(token, [])
        bboxes = []
        for d in dets:
            bb = make_mctrack_bbox(d, frame_id=fid)
            if bb is not None:
                bboxes.append(bb)

        frame_info = SunLakesFrameInfo(frame_id=fid, token=token, bboxes=bboxes, transform_matrix=None)
        trajs = tracker.track_single_frame(frame_info)

        # convert MCTrack output -> your eval format
        annos = []
        for tid, bbox in trajs.items():
            # MCTrack uses lwh; output wants wlh
            l, w, h = bbox.lwh
            size_wlh = [float(w), float(l), float(h)]
            annos.append({
                "sample_token": token,
                "translation": [float(bbox.global_xyz[0]), float(bbox.global_xyz[1]), float(bbox.global_xyz[2])],
                "size": size_wlh,
                "rotation": [float(x) for x in bbox.global_orientation],
                "velocity": [float(bbox.global_velocity[0]), float(bbox.global_velocity[1])],
                "tracking_id": str(tid),
                "tracking_name": str(bbox.category),
                "tracking_score": float(bbox.det_score),
            })

        out["results"][token] = annos

    out["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    track_path = work_dir / "tracking_result.json"
    with open(track_path, "w") as f:
        json.dump(out, f)

    c = Counter()
    for tok, lst in out["results"].items():
        for p in lst:
            c[p["tracking_name"]] += 1
    print("[TRACK] per-class counts:", dict(c))
    print("[TRACK] saved:", track_path)
    return track_path


# ============================================================
# Part B: Your AMOTA/AMOTP + MOTA/MOTP evaluator (integrated)
# ============================================================
def load_gt_pkl_by_scene(data_root: Path, split: str):
    """
    Load GT from sunlakes_infos_{split}.pkl
    Return:
      by_scene[scene] = list of frames sorted by timestamp, each:
        { token, timestamp, gt_names(np obj), gt_xy(N,2), gt_tid(list) }
    """
    pkl_path = data_root / f"sunlakes_infos_{split}.pkl"
    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)

    by_scene = defaultdict(list)
    for info in infos:
        scene = info.get("scene_id", "scene0")
        ts = float(info["timestamp"])
        token = info["token"]

        gt_names = np.array(info["gt_names"], dtype=object)
        gt_boxes = np.array(info["gt_boxes"], dtype=np.float32)  # [N,7] (x,y,z,w,l,h,yaw) or similar
        gt_ids = list(info.get("tracking_id", []))

        # safety
        if len(gt_ids) != gt_boxes.shape[0]:
            gt_ids = [f"{scene}_dummy_{i}" for i in range(gt_boxes.shape[0])]

        by_scene[scene].append({
            "token": token,
            "timestamp": ts,
            "gt_names": gt_names,
            "gt_xy": gt_boxes[:, :2].astype(np.float32),
            "gt_tid": gt_ids,
        })

    for s in by_scene:
        by_scene[s].sort(key=lambda x: x["timestamp"])
    return by_scene


def load_pred_tracking_json(tracking_json_path: Path):
    with open(tracking_json_path, "r") as f:
        data = json.load(f)
    return data["results"] if "results" in data else data


def hungarian_match(gt_xy, pred_xy, dist_th: float):
    if gt_xy.shape[0] == 0 or pred_xy.shape[0] == 0:
        return []
    diff = gt_xy[:, None, :] - pred_xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    cost = dist.copy()
    cost[cost > dist_th] = 1e18
    gi, pi = linear_sum_assignment(cost)
    matches = []
    for g, p in zip(gi, pi):
        if cost[g, p] < 1e17:
            matches.append((int(g), int(p), float(dist[g, p])))
    return matches


def eval_one_class(by_scene_gt, preds_by_token, cls_name, dist_th, score_th, max_switch_time):
    TP = FP = FN = IDS = 0
    dist_sum = 0.0
    match_cnt = 0
    P = 0
    last_match_pred = {}  # gt_tid -> (pred_tid, frame_index)

    for scene, frames in by_scene_gt.items():
        for fi, fr in enumerate(frames):
            token = fr["token"]

            gt_mask = (fr["gt_names"] == cls_name)
            gt_xy = fr["gt_xy"][gt_mask]
            gt_tid_all = np.array(fr["gt_tid"], dtype=object)
            gt_tid = gt_tid_all[gt_mask].tolist()
            P += int(gt_xy.shape[0])

            pred_list = preds_by_token.get(token, [])
            pred_keep = [
                p for p in pred_list
                if p.get("tracking_name") == cls_name and float(p.get("tracking_score", 0.0)) >= score_th
            ]

            pred_xy = np.array([pp["translation"][:2] for pp in pred_keep], dtype=np.float32) \
                if len(pred_keep) > 0 else np.zeros((0, 2), dtype=np.float32)
            pred_tid = [str(pp["tracking_id"]) for pp in pred_keep]

            matches = hungarian_match(gt_xy, pred_xy, dist_th)

            TP_f = len(matches)
            FP_f = int(pred_xy.shape[0] - TP_f)
            FN_f = int(gt_xy.shape[0] - TP_f)

            TP += TP_f
            FP += FP_f
            FN += FN_f

            for g_idx, p_idx, d in matches:
                dist_sum += d
                match_cnt += 1

                gt_id = gt_tid[g_idx]
                pr_id = pred_tid[p_idx]

                if gt_id in last_match_pred:
                    last_pr, last_fi = last_match_pred[gt_id]
                    if (fi - last_fi) <= max_switch_time and pr_id != last_pr:
                        IDS += 1
                last_match_pred[gt_id] = (pr_id, fi)

    return {"P": P, "TP": TP, "FP": FP, "FN": FN, "IDS": IDS, "dist_sum": dist_sum, "match_cnt": match_cnt}


def compute_mota_motp_at_threshold(by_scene_gt, preds_by_token, cls_name,
                                   dist_th, score_th, max_switch_time):
    st = eval_one_class(by_scene_gt, preds_by_token, cls_name, dist_th, score_th, max_switch_time)
    P = st["P"]
    if P == 0:
        return {"mota": 0.0, "motp": dist_th, "note": "No GT for this class."}

    mota = 1.0 - (st["FN"] + st["FP"] + st["IDS"]) / float(P)
    mota = float(max(0.0, mota))
    motp = float(st["dist_sum"] / st["match_cnt"]) if st["match_cnt"] > 0 else dist_th

    return {
        "mota": mota,
        "motp": motp,
        "P": int(P),
        "TP": int(st["TP"]), "FP": int(st["FP"]), "FN": int(st["FN"]), "IDS": int(st["IDS"]),
        "matches": int(st["match_cnt"]),
    }


def compute_amota_amotp(by_scene_gt, preds_by_token, cls_name, dist_th,
                        num_thresholds, min_recall, max_switch_time, pbar_desc=None):
    # gather scores for this class
    scores = []
    for token, plist in preds_by_token.items():
        for p in plist:
            if p.get("tracking_name") == cls_name:
                scores.append(float(p.get("tracking_score", 0.0)))

    if len(scores) == 0:
        return {"amota": 0.0, "amotp": dist_th, "note": "No predictions for this class."}

    scores = np.unique(np.array(scores, dtype=np.float32))
    scores.sort()
    cand_thresholds = scores[::-1]  # high -> low

    curve = []
    it = cand_thresholds
    if pbar_desc is not None:
        it = tqdm(cand_thresholds, desc=pbar_desc, unit="th", leave=False)

    for th in it:
        st = eval_one_class(by_scene_gt, preds_by_token, cls_name, dist_th, float(th), max_switch_time)
        P = st["P"]
        if P == 0:
            continue
        recall = st["TP"] / max(1, P)
        r = max(float(recall), 1e-12)
        motar = 1.0 - (st["IDS"] + st["FP"] + st["FN"] - (1.0 - r) * P) / (r * P)
        motar = max(0.0, float(motar))
        motp = (st["dist_sum"] / st["match_cnt"]) if st["match_cnt"] > 0 else dist_th
        curve.append({"recall": float(recall), "motar": motar, "motp": float(motp)})

    if len(curve) == 0:
        return {"amota": 0.0, "amotp": dist_th, "note": "Empty curve (no GT or no matches)."}

    recall_levels = (np.arange(1, num_thresholds + 1, dtype=np.float32) / float(num_thresholds))
    recall_levels = recall_levels[recall_levels >= float(min_recall)]

    motar_list, motp_list = [], []
    for r in recall_levels:
        feasible = [pt for pt in curve if pt["recall"] >= float(r) - 1e-12]
        if not feasible:
            motar_list.append(0.0)
            motp_list.append(dist_th)
            continue
        feasible.sort(key=lambda x: (x["motar"], -x["motp"]), reverse=True)
        best = feasible[0]
        motar_list.append(best["motar"])
        motp_list.append(best["motp"])

    return {
        "amota": float(np.mean(motar_list)) if motar_list else 0.0,
        "amotp": float(np.mean(motp_list)) if motp_list else dist_th,
        "num_recall_levels": int(len(recall_levels)),
        "curve_points": int(len(curve)),
    }


def run_eval(data_root: Path,
             split: str,
             tracking_json_path: Path,
             work_dir: Path,
             dist_th: float,
             num_thresholds: int,
             min_recall: float,
             max_switch_time: int,
             mota_score_th: float):
    by_scene_gt = load_gt_pkl_by_scene(data_root, split)
    preds_by_token = load_pred_tracking_json(tracking_json_path)

    # token sanity
    gt_tokens = set()
    for s, frames in by_scene_gt.items():
        for fr in frames:
            gt_tokens.add(fr["token"])
    pred_tokens = set(preds_by_token.keys())
    print(f"[EVAL] GT tokens={len(gt_tokens)} pred tokens={len(pred_tokens)} intersection={len(gt_tokens & pred_tokens)}")

    per_class = {}
    amota_all, amotp_all = [], []
    mota_all, motp_all = [], []

    for cls in tqdm(SUNLAKES_TRACKING_NAMES, desc="[EVAL] Classes", unit="class"):
        am_res = compute_amota_amotp(
            by_scene_gt, preds_by_token, cls,
            dist_th=dist_th,
            num_thresholds=num_thresholds,
            min_recall=min_recall,
            max_switch_time=max_switch_time,
            pbar_desc=f"[EVAL] {cls} thresholds"
        )
        m_res = compute_mota_motp_at_threshold(
            by_scene_gt, preds_by_token, cls,
            dist_th=dist_th,
            score_th=mota_score_th,
            max_switch_time=max_switch_time
        )

        per_class[cls] = {"amota_amotp": am_res, "mota_motp": m_res}

        amota_all.append(am_res["amota"])
        amotp_all.append(am_res["amotp"])
        mota_all.append(m_res["mota"])
        motp_all.append(m_res["motp"])

        note = ""
        if "note" in am_res: note += f" | AM-note={am_res['note']}"
        if "note" in m_res:  note += f" | M-note={m_res['note']}"

        tqdm.write(
            f"[{cls:>20}] "
            f"AMOTA={am_res['amota']:.4f} AMOTP={am_res['amotp']:.4f} || "
            f"MOTA@{mota_score_th:.3f}={m_res['mota']:.4f} MOTP={m_res['motp']:.4f}"
            f"{note}"
        )

    overall = {
        "mean_amota": float(np.mean(amota_all)) if amota_all else 0.0,
        "mean_amotp": float(np.mean(amotp_all)) if amotp_all else dist_th,
        "dist_th": dist_th,
        "num_thresholds": num_thresholds,
        "min_recall": min_recall,
        "max_switch_time": max_switch_time,
        "mean_mota": float(np.mean(mota_all)) if mota_all else 0.0,
        "mean_motp": float(np.mean(motp_all)) if motp_all else dist_th,
        "mota_score_th": mota_score_th
    }

    print("\n========== OVERALL ==========")
    print(f"mean AMOTA = {overall['mean_amota']:.4f}")
    print(f"mean AMOTP = {overall['mean_amotp']:.4f}")
    print(f"mean MOTA@{mota_score_th:.3f} = {overall['mean_mota']:.4f}")
    print(f"mean MOTP           = {overall['mean_motp']:.4f}")

    out = {"per_class": per_class, "overall": overall}
    out_path = work_dir / f"amota_amotp_{split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[OK] saved eval to: {out_path}")
    return out_path


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser("MCTrack adapted to SunLakes format + AMOTA eval")
    p.add_argument("--cfg_path", type=str, default="config/sunlakes.yaml", help="MCTrack yaml config")
    p.add_argument("--data_root", type=str, required=True, help="SunLakes root containing sunlakes_infos_{split}.pkl")
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--pred_json", type=str, required=True, help="detection json (results_nusc.json-like)")
    p.add_argument("--work_dir", type=str, default="./results")

    # eval params
    p.add_argument("--dist_th", type=float, default=2.0)
    p.add_argument("--num_thresholds", type=int, default=40)
    p.add_argument("--min_recall", type=float, default=0.1)
    p.add_argument("--max_switch_time", type=int, default=999999)
    p.add_argument("--mota_score_th", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.load(open(args.cfg_path, "r"), Loader=yaml.Loader)

    t0 = time.time()
    track_path = run_tracking(
        cfg=cfg,
        data_root=Path(args.data_root),
        split=args.split,
        pred_json=Path(args.pred_json),
        work_dir=work_dir,
    )

 
    run_eval(
            data_root=Path(args.data_root),
            split=args.split,
            tracking_json_path=track_path,
            work_dir=work_dir,
            dist_th=args.dist_th,
            num_thresholds=args.num_thresholds,
            min_recall=args.min_recall,
            max_switch_time=args.max_switch_time,
            mota_score_th=args.mota_score_th,
    )

    print(f"Elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()