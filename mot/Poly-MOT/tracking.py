# -*- coding: utf-8 -*-

import os
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor, as_completed


# -------------------------
# IMPORTANT: import your Poly-MOT tracker and preprocessing utils from your repo
# -------------------------
import yaml
from tracking.nusc_tracker import Tracker  # your Poly-MOT Tracker
from utils.io import load_file
from pre_processing import dictdet2array, arraydet2box
from pre_processing import blend_nms  # ensure NMS functions exist in globals()


# -------------------------
# SunLakes classes & gating (for eval script compatibility; tracking itself uses Poly-MOT cfg)
# -------------------------
SUNLAKES_TRACKING_NAMES = [
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck'
]


# =========================
# 1) SunLakes Loader (Poly-MOT compatible)
# =========================
class SunLakesloader:
    """
    Output data_info fields are identical to your NuScenesloader:
    {
        'is_first_frame': bool
        'timestamp': int
        'sample_token': str
        'seq_id': int
        'frame_id': int
        'has_velo': bool
        'np_dets': np.array, [det_num, 14]
        'np_dets_bottom_corners': np.array, [det_num, 4, 2]
        'box_dets': np.array[NuscBox], [det_num]
        'no_dets': bool
        'det_num': int
    }
    """

    def __init__(self, data_root, split, pred_json, config, pred_token_key="results"):
        self.data_root = Path(data_root)
        self.split = split
        self.config = config

        # 1) build frame order from pkl (by scene_id then timestamp)
        pkl_path = self.data_root / f"sunlakes_infos_{split}.pkl"
        with open(pkl_path, "rb") as f:
            infos = pickle.load(f)

        by_scene = defaultdict(list)
        for info in infos:
            scene = info.get("scene_id", "scene0")
            ts = float(info["timestamp"])
            token = info["token"]
            by_scene[scene].append((ts, token))

        self.frames = []
        for seq_id, (scene_id, seq) in enumerate(by_scene.items()):
            seq.sort(key=lambda x: x[0])
            for k, (ts, token) in enumerate(seq):
                self.frames.append({
                    "token": token,
                    "scene_id": scene_id,
                    "seq_id": seq_id + 1,     # NuScenesloader seq_id increments from 1
                    "frame_id": k + 1,        # NuScenesloader first frame_id=1
                    "is_first_frame": (k == 0),
                })

        self.all_sample_token = [fr["token"] for fr in self.frames]

        # 2) load detector json
        pred_all = load_file(pred_json) if pred_json.endswith((".json", ".JSON")) else load_file(pred_json)
        self.detector = pred_all[pred_token_key]  # token -> list[det]

        # 3) preprocessing cfg (reuse Poly-MOT cfg)
        self.SF_thre = config['preprocessing']['SF_thre']      # indexed by class_label int
        self.NMS_thre = config['preprocessing']['NMS_thre']
        self.NMS_type = config['preprocessing']['NMS_type']    # e.g. "blend_nms"
        self.NMS_metric = config['preprocessing']['NMS_metric']

    def __getitem__(self, item) -> dict:
        fr = self.frames[item]
        curr_token = fr["token"]
        ori_dets = self.detector.get(curr_token, [])

        # dictdet2array expects nuScenes detection fields:
        # translation,size,velocity,rotation,detection_score,detection_name
        list_dets, np_dets = dictdet2array(
            ori_dets,
            'translation', 'size', 'velocity', 'rotation',
            'detection_score', 'detection_name'
        )

        # Score Filter: det[-2]=score, det[-1]=class_label (int)
        if len(list_dets) > 0:
            np_dets = np.array([det for det in list_dets if det[-2] > self.SF_thre[int(det[-1])]])
        else:
            np_dets = np.zeros((0,), dtype=np.float32)

        # NMS
        if len(np_dets) != 0:
            box_dets, np_dets_bottom_corners = arraydet2box(np_dets)
            tmp_infos = {'np_dets': np_dets, 'np_dets_bottom_corners': np_dets_bottom_corners}
            keep = globals()[self.NMS_type](box_infos=tmp_infos, metrics=self.NMS_metric, thre=self.NMS_thre)
            keep_num = len(keep)
        else:
            keep = keep_num = 0
            box_dets = np.zeros(0)
            np_dets_bottom_corners = np.zeros(0)

        data_info = {
            'is_first_frame': fr["is_first_frame"],
            'timestamp': item,
            'sample_token': curr_token,
            'seq_id': fr["seq_id"],
            'frame_id': fr["frame_id"],
            'has_velo': self.config['basic']['has_velo'],
            'np_dets': np_dets[keep] if keep_num != 0 else np.zeros(0),
            'np_dets_bottom_corners': np_dets_bottom_corners[keep] if keep_num != 0 else np.zeros(0),
            'box_dets': box_dets[keep] if keep_num != 0 else np.zeros(0),
            'no_dets': keep_num == 0,
            'det_num': keep_num,
        }
        return data_info

    def __len__(self):
        return len(self.all_sample_token)


# =========================
# 2) Export Poly-MOT output -> tracking_result.json
# =========================
def export_tracking_result(frames_loader: SunLakesloader, tracker: Tracker, out_path: Path):
    out = {"results": {}, "meta": {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }}

    cls_counter = Counter()

    for i in tqdm(range(len(frames_loader)), desc="[PolyMOT] Tracking", unit="frame"):
        frame_data = frames_loader[i]
        token = frame_data['sample_token']

        tracker.tracking(frame_data)  # in-place update

        if 'no_val_track_result' in frame_data:
            out["results"][token] = []
            continue

        annos = []
        for pb in frame_data['box_track_res']:
            # orientation may be Quaternion or array
            rot = pb.orientation.elements.tolist() if hasattr(pb.orientation, "elements") else list(pb.orientation)

            vel = [0.0, 0.0]
            if hasattr(pb, "velocity") and pb.velocity is not None:
                vel = [float(pb.velocity[0]), float(pb.velocity[1])]

            ann = {
                "sample_token": token,
                "translation": [float(pb.center[0]), float(pb.center[1]), float(pb.center[2])],
                "size": [float(pb.wlh[0]), float(pb.wlh[1]), float(pb.wlh[2])],
                "rotation": [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])],
                "velocity": vel,
                "tracking_id": str(getattr(pb, "tracking_id", getattr(pb, "track_id", 0))),
                "tracking_name": str(getattr(pb, "name", "car")),
                "tracking_score": float(getattr(pb, "score", 0.0)),
            }
            annos.append(ann)
            cls_counter[ann["tracking_name"]] += 1

        out["results"][token] = annos

    with open(out_path, "w") as f:
        json.dump(out, f)

    print("[TRACK] per-class counts:", dict(cls_counter))
    print("[TRACK] saved:", str(out_path))
    return out_path


# =========================
# 3) AMOTA/AMOTP eval (integrated from your script)
# =========================
def load_gt_pkl(data_root: Path, split: str):
    pkl_path = data_root / f"sunlakes_infos_{split}.pkl"
    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)

    by_scene = defaultdict(list)
    for info in infos:
        scene = info.get("scene_id", "scene0")
        ts = float(info["timestamp"])
        token = info["token"]

        gt_names = np.array(info["gt_names"], dtype=object)
        gt_boxes = np.array(info["gt_boxes"], dtype=np.float32)
        gt_ids = list(info.get("tracking_id", []))

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


def load_pred_tracking_json(pred_path: Path):
    with open(pred_path, "r") as f:
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
            pred_keep = [p for p in pred_list
                         if p.get("tracking_name") == cls_name and float(p.get("tracking_score", 0.0)) >= score_th]

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
    st = eval_one_class(by_scene_gt, preds_by_token, cls_name,
                        dist_th=dist_th, score_th=score_th, max_switch_time=max_switch_time)
    P = st["P"]
    if P == 0:
        return {"mota": 0.0, "motp": dist_th, "note": "No GT for this class."}

    mota = 1.0 - (st["FN"] + st["FP"] + st["IDS"]) / float(P)
    mota = float(max(0.0, mota))
    motp = float(st["dist_sum"] / st["match_cnt"]) if st["match_cnt"] > 0 else dist_th

    return {
        "mota": mota, "motp": motp,
        "P": int(P),
        "TP": int(st["TP"]), "FP": int(st["FP"]), "FN": int(st["FN"]), "IDS": int(st["IDS"]),
        "matches": int(st["match_cnt"]),
    }


def compute_amota_amotp(by_scene_gt, preds_by_token, cls_name, dist_th, num_thresholds, min_recall, max_switch_time):
    scores = []
    for token, plist in preds_by_token.items():
        for p in plist:
            if p.get("tracking_name") == cls_name:
                scores.append(float(p.get("tracking_score", 0.0)))
    if len(scores) == 0:
        return {"amota": 0.0, "amotp": dist_th, "note": "No predictions for this class."}

    scores = np.unique(np.array(scores, dtype=np.float32))
    scores.sort()
    cand_thresholds = scores[::-1]

    curve = []
    for th in cand_thresholds:
        st = eval_one_class(by_scene_gt, preds_by_token, cls_name, dist_th, float(th), max_switch_time)
        P = st["P"]
        if P == 0:
            continue
        recall = st["TP"] / max(1, P)
        r = max(recall, 1e-12)
        motar = 1.0 - (st["IDS"] + st["FP"] + st["FN"] - (1.0 - r) * P) / (r * P)
        motar = max(0.0, float(motar))
        motp = (st["dist_sum"] / st["match_cnt"]) if st["match_cnt"] > 0 else dist_th
        curve.append({"recall": float(recall), "motar": motar, "motp": motp})

    if len(curve) == 0:
        return {"amota": 0.0, "amotp": dist_th, "note": "Empty curve (no GT or no matches)."}

    recall_levels = (np.arange(1, num_thresholds + 1, dtype=np.float32) / float(num_thresholds))
    recall_levels = recall_levels[recall_levels >= float(min_recall)]

    motar_list, motp_list = [], []
    for r in recall_levels:
        feasible = [pt for pt in curve if pt["recall"] >= r - 1e-12]
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


def run_eval(data_root: str, split: str, tracking_json_path: Path, workers=10,
             dist_th=2.0, num_thresholds=40, min_recall=0.1, max_switch_time=999999, mota_score_th=0.5,
             class_names=None, out_path=None):
    if class_names is None:
        class_names = SUNLAKES_TRACKING_NAMES

    by_scene_gt = load_gt_pkl(Path(data_root), split)
    preds_by_token = load_pred_tracking_json(tracking_json_path)

    gt_tokens = set()
    for frames in by_scene_gt.values():
        for fr in frames:
            gt_tokens.add(fr["token"])
    pred_tokens = set(preds_by_token.keys())
    print(f"[EVAL] GT tokens={len(gt_tokens)} pred tokens={len(pred_tokens)} intersection={len(gt_tokens & pred_tokens)}")

    per_class = {}
    amota_all, amotp_all = [], []
    mota_all, motp_all = [], []

    def _eval_one_cls(cls):
        am_res = compute_amota_amotp(by_scene_gt, preds_by_token, cls,
                                 dist_th, num_thresholds, min_recall, max_switch_time)
        m_res = compute_mota_motp_at_threshold(by_scene_gt, preds_by_token, cls,
                                           dist_th, mota_score_th, max_switch_time)
        return cls, am_res, m_res

    max_workers = min(len(class_names), int(os.cpu_count() or 8), int(workers))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_eval_one_cls, cls): cls for cls in class_names}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="[EVAL] Classes", unit="class"):
            cls, am_res, m_res = fut.result()

            per_class[cls] = {"amota_amotp": am_res, "mota_motp": m_res}
            amota_all.append(am_res["amota"])
            amotp_all.append(am_res["amotp"])
            mota_all.append(m_res["mota"])
            motp_all.append(m_res["motp"])

            note = ""
            if "note" in am_res: note += f" | AM-note={am_res['note']}"
            if "note" in m_res:  note += f" | M-note={m_res['note']}"
            tqdm.write(
                f"[{cls:>20}] AMOTA={am_res['amota']:.4f} AMOTP={am_res['amotp']:.4f} || "
                f"MOTA@{mota_score_th:.3f}={m_res['mota']:.4f} MOTP={m_res['motp']:.4f}{note}"
            )

    overall = {
        "mean_amota": float(np.mean(amota_all)) if amota_all else 0.0,
        "mean_amotp": float(np.mean(amotp_all)) if amotp_all else dist_th,
        "mean_mota": float(np.mean(mota_all)) if mota_all else 0.0,
        "mean_motp": float(np.mean(motp_all)) if motp_all else dist_th,
        "dist_th": dist_th,
        "num_thresholds": num_thresholds,
        "min_recall": min_recall,
        "max_switch_time": max_switch_time,
        "mota_score_th": mota_score_th
    }

    print("\n========== OVERALL ==========")
    print(f"mean AMOTA = {overall['mean_amota']:.4f}")
    print(f"mean AMOTP = {overall['mean_amotp']:.4f}")
    print(f"mean MOTA@{mota_score_th:.3f} = {overall['mean_mota']:.4f}")
    print(f"mean MOTP           = {overall['mean_motp']:.4f}")

    out = {"per_class": per_class, "overall": overall}
    if out_path is None:
        out_path = tracking_json_path.parent / f"amota_amotp_{split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("[OK] saved eval to:", str(out_path))
    return out_path


# =========================
# 4) Main
# =========================
def parse_args():
    p = argparse.ArgumentParser("Poly-MOT on SunLakes (all-in-one)")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train","val"])
    p.add_argument("--pred_json", type=str, required=True)
    p.add_argument("--config_path", type=str, default="./config/sunlakes_config.yaml")
    p.add_argument("--workers", type=int, default=10, help="threads for eval/preprocess")
    p.add_argument("--work_dir", type=str, default="./results")

    # eval options
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

    cfg = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)

    loader = SunLakesloader(
        data_root=args.data_root,
        split=args.split,
        pred_json=args.pred_json,
        config=cfg,
    )

    tracker = Tracker(config=cfg)
    track_path = work_dir / "tracking_result.json"

    export_tracking_result(loader, tracker, track_path)


    run_eval(
            data_root=args.data_root,
            split=args.split,
            tracking_json_path=track_path,
            workers=args.workers,
            dist_th=args.dist_th,
            num_thresholds=args.num_thresholds,
            min_recall=args.min_recall,
            max_switch_time=args.max_switch_time,
            mota_score_th=args.mota_score_th,
            class_names=SUNLAKES_TRACKING_NAMES,
            out_path=work_dir / f"amota_amotp_{args.split}.json"
    )


if __name__ == "__main__":
    main()
