#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import argparse
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict, Counter
from xinshuo_io import mkdir_if_missing
import os
import numpy as np

from AB3DMOT_libs.model import AB3DMOT


SUNLAKES_TRACKING_NAMES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
]

# 每类独立参数：你可以继续按 SunLakes 分布调
SUNLAKES_AB3DMOT_PARAMS = {
    "car": dict(
        algm="greedy", metric="giou_3d", thres=-0.2, min_hits=1, max_age=2
    ),
    "pedestrian": dict(
        algm="greedy", metric="dist_3d", thres=1.0, min_hits=1, max_age=2
    ),
    "truck": dict(
        algm="greedy", metric="giou_3d", thres=-0.2, min_hits=1, max_age=2
    ),
    "trailer": dict(
        algm="greedy", metric="giou_3d", thres=-0.2, min_hits=3, max_age=2
    ),
    "bus": dict(
        algm="greedy", metric="giou_3d", thres=-0.2, min_hits=1, max_age=2
    ),
    "motorcycle": dict(
        algm="greedy", metric="giou_3d", thres=-0.4, min_hits=3, max_age=2
    ),
    "bicycle": dict(
        algm="greedy", metric="giou_3d", thres=-0.2, min_hits=3, max_age=2
    ),

    # SunLakes extra
    "construction_vehicle": dict(
        algm="greedy", metric="giou_3d", thres=-0.2, min_hits=1, max_age=2
    ),
    "traffic_cone": dict(
        algm="greedy", metric="giou_3d", thres=-0.2, min_hits=1, max_age=2
    ),
    "barrier": dict(
        algm="greedy", metric="giou_3d", thres=-0.2, min_hits=1, max_age=2
    ),
}

class DummyLog:
    def write(self, s):
        pass
    def flush(self):
        pass


DEFAULT_PARAMS = dict(algm="greedy", metric="dist_3d", thres=2.0, min_hits=1, max_age=2)

# =========================
# Eval: AMOTA/AMOTP + MOTA/MOTP
# =========================

def load_gt_pkl_for_eval(data_root: Path, split: str):
    """
    从 sunlakes_infos_{split}.pkl 读取：
      - token
      - gt_names
      - gt_boxes[:, :2] 作为匹配中心 (x,y)
      - tracking_id 作为 GT track id（用于 IDS）
    返回：by_scene: scene -> list(frames sorted by timestamp)
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
        gt_boxes = np.array(info["gt_boxes"], dtype=np.float32)
        gt_ids = list(info.get("tracking_id", []))

        # safety: id 数量不对就造 dummy（否则 IDS 失真，但不会 crash）
        if len(gt_ids) != gt_boxes.shape[0]:
            gt_ids = [f"{scene}_dummy_{i}" for i in range(gt_boxes.shape[0])]

        by_scene[scene].append({
            "token": token,
            "timestamp": ts,
            "gt_names": gt_names,
            "gt_xy": gt_boxes[:, :2].astype(np.float32),
            "gt_tid": np.array(gt_ids, dtype=object),
        })

    for s in by_scene:
        by_scene[s].sort(key=lambda x: x["timestamp"])
    return by_scene


def load_pred_tracking_json_for_eval(pred_path: Path):
    with open(pred_path, "r") as f:
        data = json.load(f)
    preds_by_token = data["results"] if "results" in data else data
    return preds_by_token


def hungarian_match_xy(gt_xy: np.ndarray, pred_xy: np.ndarray, dist_th: float):
    """
    gt_xy: (N,2), pred_xy:(M,2)
    返回 matches: list[(g_idx, p_idx, dist)]
    """
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


def eval_one_class_at_score(by_scene_gt, preds_by_token, cls_name: str,
                            dist_th: float, score_th: float, max_switch_time: int):
    """
    固定 score_th 下统计：P, TP, FP, FN, IDS, dist_sum, match_cnt
    """
    cls = str(cls_name).lower()

    TP = FP = FN = IDS = 0
    dist_sum = 0.0
    match_cnt = 0
    P = 0

    # gt_tid -> (last_pred_tid, last_frame_index)
    last_match_pred = {}

    for scene, frames in by_scene_gt.items():
        for fi, fr in enumerate(frames):
            token = fr["token"]

            # GT filter
            gt_names = np.char.lower(fr["gt_names"].astype(str))
            gt_mask = (gt_names == cls)
            gt_xy = fr["gt_xy"][gt_mask]
            gt_tid = fr["gt_tid"][gt_mask].tolist()
            P += int(gt_xy.shape[0])

            # Pred filter
            plist = preds_by_token.get(token, [])
            pred_keep = []
            for p in plist:
                pn = str(p.get("tracking_name", "")).lower()
                if pn != cls:
                    continue
                if float(p.get("tracking_score", 0.0)) < float(score_th):
                    continue
                pred_keep.append(p)

            pred_xy = np.array([pp["translation"][:2] for pp in pred_keep], dtype=np.float32) \
                      if len(pred_keep) > 0 else np.zeros((0, 2), dtype=np.float32)
            pred_tid = [str(pp.get("tracking_id")) for pp in pred_keep]

            matches = hungarian_match_xy(gt_xy, pred_xy, dist_th)

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

    return {
        "P": P, "TP": TP, "FP": FP, "FN": FN, "IDS": IDS,
        "dist_sum": dist_sum, "match_cnt": match_cnt
    }


def compute_mota_motp(by_scene_gt, preds_by_token, cls_name: str,
                      dist_th: float, score_th: float, max_switch_time: int):
    """
    固定阈值单点：MOTA/MOTP
    """
    st = eval_one_class_at_score(by_scene_gt, preds_by_token, cls_name,
                                 dist_th=dist_th, score_th=score_th, max_switch_time=max_switch_time)
    P = st["P"]
    if P == 0:
        return {"mota": 0.0, "motp": dist_th, "note": "No GT for this class."}

    mota = 1.0 - (st["FN"] + st["FP"] + st["IDS"]) / float(P)
    mota = float(max(0.0, mota))  # 常见做法：下限裁剪

    motp = float(st["dist_sum"] / st["match_cnt"]) if st["match_cnt"] > 0 else float(dist_th)

    return {
        "mota": mota, "motp": motp,
        "P": int(P), "TP": int(st["TP"]), "FP": int(st["FP"]), "FN": int(st["FN"]), "IDS": int(st["IDS"]),
        "matches": int(st["match_cnt"])
    }


def compute_amota_amotp(by_scene_gt, preds_by_token, cls_name: str,
                        dist_th: float, num_thresholds: int, min_recall: float, max_switch_time: int):
    """
    扫所有 unique score 作为阈值点，形成 curve，然后按 recall_levels 平均。
    与你之前实现一致：每个 recall level 取 recall>=r 的点里 motar 最大者。
    """
    cls = str(cls_name).lower()

    # gather all scores for this class
    scores = []
    for tok, plist in preds_by_token.items():
        for p in plist:
            if str(p.get("tracking_name", "")).lower() == cls:
                scores.append(float(p.get("tracking_score", 0.0)))

    if len(scores) == 0:
        return {"amota": 0.0, "amotp": float(dist_th), "note": "No predictions for this class."}

    scores = np.unique(np.array(scores, dtype=np.float32))
    scores.sort()
    cand_thresholds = scores[::-1]  # high -> low

    curve = []
    for th in cand_thresholds:
        st = eval_one_class_at_score(by_scene_gt, preds_by_token, cls,
                                     dist_th=dist_th, score_th=float(th), max_switch_time=max_switch_time)
        P = st["P"]
        if P == 0:
            continue

        recall = st["TP"] / max(1, P)
        r = max(float(recall), 1e-12)

        # nuScenes-like MOTAR (your previous formula)
        motar = 1.0 - (st["IDS"] + st["FP"] + st["FN"] - (1.0 - r) * P) / (r * P)
        motar = max(0.0, float(motar))

        motp = (st["dist_sum"] / st["match_cnt"]) if st["match_cnt"] > 0 else float(dist_th)

        curve.append({
            "score_th": float(th),
            "recall": float(recall),
            "motar": float(motar),
            "motp": float(motp),
            "TP": int(st["TP"]), "FP": int(st["FP"]), "FN": int(st["FN"]), "IDS": int(st["IDS"])
        })

    if len(curve) == 0:
        return {"amota": 0.0, "amotp": float(dist_th), "note": "Empty curve (no GT or no matches)."}

    # recall levels
    recall_levels = (np.arange(1, int(num_thresholds) + 1, dtype=np.float32) / float(num_thresholds))
    recall_levels = recall_levels[recall_levels >= float(min_recall)]

    motar_list, motp_list = [], []
    for rlv in recall_levels:
        feasible = [pt for pt in curve if pt["recall"] >= float(rlv) - 1e-12]
        if not feasible:
            motar_list.append(0.0)
            motp_list.append(float(dist_th))
            continue
        # choose best motar; tie-break motp smaller better
        feasible.sort(key=lambda x: (x["motar"], -x["motp"]), reverse=True)
        best = feasible[0]
        motar_list.append(best["motar"])
        motp_list.append(best["motp"])

    return {
        "amota": float(np.mean(motar_list)) if motar_list else 0.0,
        "amotp": float(np.mean(motp_list)) if motp_list else float(dist_th),
        "num_recall_levels": int(len(recall_levels)),
        "curve_points": int(len(curve)),
        # 你想debug的话可以把 curve 也存出去（很大，默认不存）
        # "curve": curve
    }


def run_eval(args, tracking_json_path: Path):
    data_root = Path(args.data_root)
    by_scene_gt = load_gt_pkl_for_eval(data_root, args.split)
    preds_by_token = load_pred_tracking_json_for_eval(tracking_json_path)

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

    for cls in SUNLAKES_TRACKING_NAMES:
        am = compute_amota_amotp(
            by_scene_gt, preds_by_token, cls,
            dist_th=args.dist_th,
            num_thresholds=args.num_thresholds,
            min_recall=args.min_recall,
            max_switch_time=args.max_switch_time
        )
        m = compute_mota_motp(
            by_scene_gt, preds_by_token, cls,
            dist_th=args.dist_th,
            score_th=args.mota_score_th,
            max_switch_time=args.max_switch_time
        )

        per_class[cls] = {"amota_amotp": am, "mota_motp": m}
        amota_all.append(am["amota"])
        amotp_all.append(am["amotp"])
        mota_all.append(m["mota"])
        motp_all.append(m["motp"])

        note = ""
        if "note" in am: note += f" | AM-note={am['note']}"
        if "note" in m:  note += f" | M-note={m['note']}"

        print(
            f"[{cls:>20}] "
            f"AMOTA={am['amota']:.4f} AMOTP={am['amotp']:.4f} || "
            f"MOTA@{args.mota_score_th:.3f}={m['mota']:.4f} MOTP={m['motp']:.4f}"
            f"{note}"
        )

    overall = {
        "mean_amota": float(np.mean(amota_all)) if amota_all else 0.0,
        "mean_amotp": float(np.mean(amotp_all)) if amotp_all else float(args.dist_th),
        "mean_mota": float(np.mean(mota_all)) if mota_all else 0.0,
        "mean_motp": float(np.mean(motp_all)) if motp_all else float(args.dist_th),
        "dist_th": float(args.dist_th),
        "num_thresholds": int(args.num_thresholds),
        "min_recall": float(args.min_recall),
        "max_switch_time": int(args.max_switch_time),
        "mota_score_th": float(args.mota_score_th),
    }

    print("\n========== OVERALL ==========")
    print(f"mean AMOTA = {overall['mean_amota']:.4f}")
    print(f"mean AMOTP = {overall['mean_amotp']:.4f}")
    print(f"mean MOTA@{args.mota_score_th:.3f} = {overall['mean_mota']:.4f}")
    print(f"mean MOTP           = {overall['mean_motp']:.4f}")

    out = {"per_class": per_class, "overall": overall}
    out_path = Path(args.work_dir) / f"tracking_metrics_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] saved eval to: {out_path}")
    return out_path



def quat_to_yaw(q):
    if q is None:
        return 0.0
    if isinstance(q, (list, tuple)) and len(q) == 4:
        w, x, y, z = map(float, q)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(t3, t4))
    return 0.0


def yaw_to_quat(yaw):
    yaw = float(yaw)
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


def make_cfg(det_name="custom", ego_com=False, vis=False, affi_pro=False):
    return SimpleNamespace(
        dataset="nuScenes",          # 必须是 nuScenes 才走 AB3DMOT 的 nuScenes 分支
        det_name=det_name,
        split="val",
        cat_list=list(SUNLAKES_TRACKING_NAMES),
        num_hypo=1,
        score_threshold=-10000,
        ego_com=bool(ego_com),
        vis=bool(vis),
        affi_pro=bool(affi_pro),
    )


def load_frames_from_pkl(data_root: Path, split: str):
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
    for scene_id, seq in by_scene.items():
        seq.sort(key=lambda x: x[0])
        for k, (ts, token) in enumerate(seq):
            frames.append({
                "scene_id": scene_id,
                "token": token,
                "timestamp": ts,
                "first": (k == 0),
                "frame_idx_in_scene": k,
            })
    return frames


def load_detection_json(pred_json: Path, token_key="results"):
    with open(pred_json, "r") as f:
        data = json.load(f)
    if token_key not in data:
        raise KeyError(f"pred_json missing key='{token_key}'. keys={list(data.keys())}")
    return data[token_key]


def det_list_to_ab3dmot_arrays(det_list, cls_name, score_th=0.0, size_order="wlh"):
    rows = []
    infos = []
    for d in det_list:
        if d.get("detection_name") != cls_name:
            continue
        sc = float(d.get("detection_score", 0.0))
        if sc < float(score_th):
            continue

        trans = d.get("translation", None)
        size = d.get("size", None)
        rot = d.get("rotation", None)
        if trans is None or size is None:
            continue

        x, y, z = map(float, trans[:3])

        if size_order.lower() == "wlh":
            w, l, h = map(float, size[:3])     # nuScenes: [w,l,h]
        elif size_order.lower() == "lwh":
            l, w, h = map(float, size[:3])
        elif size_order.lower() == "hwl":
            h, w, l = map(float, size[:3])
        else:
            raise ValueError(f"Unknown --size_order {size_order}. Use wlh/lwh/hwl")

        yaw = quat_to_yaw(rot)

        # AB3DMOT raw expects [h,w,l,x,y,z,theta]
        rows.append([h, w, l, x, y, z, yaw])

        # info 保证 7 维：把 score 放第一个，其余占位
        infos.append([sc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    if len(rows) == 0:
        dets = np.zeros((0, 7), dtype=np.float32)
        info = np.zeros((0, 7), dtype=np.float32)
    else:
        dets = np.asarray(rows, dtype=np.float32)
        info = np.asarray(infos, dtype=np.float32)

    return {"dets": dets, "info": info}


class SunLakesAB3DMOT(AB3DMOT):
    def get_param(self, cfg, cat):
        cat_l = str(cat).lower()
        p = SUNLAKES_AB3DMOT_PARAMS.get(cat_l, DEFAULT_PARAMS)

        algm = p["algm"]
        metric = p["metric"]
        thres = float(p["thres"])
        min_hits = int(p["min_hits"])
        max_age = int(p["max_age"])

        # dist_* is cost threshold inside AB3DMOT
        if metric in ["dist_3d", "dist_2d", "m_dis"]:
            thres *= -1.0

        self.algm = algm
        self.metric = metric
        self.thres = thres
        self.max_age = max_age
        self.min_hits = min_hits

        if self.metric in ["dist_3d", "dist_2d", "m_dis"]:
            self.max_sim, self.min_sim = 0.0, -100.0
        elif self.metric in ["iou_2d", "iou_3d"]:
            self.max_sim, self.min_sim = 1.0, 0.0
        elif self.metric in ["giou_2d", "giou_3d"]:
            self.max_sim, self.min_sim = 1.0, -1.0
        else:
            self.max_sim, self.min_sim = 1.0, -1.0

def process_one_class(cls, det_list, tracker_obj, frame_idx, scene_id, args, token):
    dets_all = det_list_to_ab3dmot_arrays(
        det_list, cls_name=cls,
        score_th=args.score_th
    )

    results, _affi = tracker_obj.track(dets_all, frame_idx, scene_id)
    res0 = results[0]
    if res0 is None or res0.shape[0] == 0:
        return []

    # build id->velocity after track updates internal KFs
    id2vel = {}
    for kf in tracker_obj.trackers:
        try:
            v = kf.get_velocity().reshape(-1)
            dx, dy = float(v[0]), float(v[1])
        except Exception:
            dx, dy = 0.0, 0.0
        id2vel[int(kf.id)] = (dx, dy)

    annos = []
    for row in res0:
        h, w, l, x, y, z, yaw = map(float, row[:7])
        tid = int(row[7])
        score = float(row[8]) if row.shape[0] > 8 else 0.0
        dx, dy = id2vel.get(tid, (0.0, 0.0))

        annos.append({
            "sample_token": token,
            "translation": [x, y, z],
            "size": [w, l, h],
            "rotation": yaw_to_quat(yaw),
            "velocity": [dx, dy],
            "tracking_id": str(tid),
            "tracking_name": cls,
            "tracking_score": score,
        })
    return annos

def parse_args():
    p = argparse.ArgumentParser("SunLakes AB3DMOT Runner (no yml/config)")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--pred_json", type=str, required=True, help="results_nusc.json (detection output)")
    p.add_argument("--work_dir", type=str, default="./results")
    p.add_argument("--score_th", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out_name", type=str, default="tracking_result.json")

    # ---------------- eval params ----------------
    p.add_argument("--dist_th", type=float, default=2.0, help="xy-center matching threshold (meters)")
    p.add_argument("--num_thresholds", type=int, default=40, help="AMOTA recall levels")
    p.add_argument("--min_recall", type=float, default=0.1, help="AMOTA min recall level")
    p.add_argument("--max_switch_time", type=int, default=999999, help="max frame gap for counting IDSW")
    p.add_argument("--mota_score_th", type=float, default=0.5, help="fixed threshold for MOTA/MOTP")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames_from_pkl(data_root, args.split)
    with open(work_dir / "frames_meta.json", "w") as f:
        json.dump({"frames": frames}, f, indent=2)

    predictions = load_detection_json(Path(args.pred_json))

    cfg = make_cfg()

    executor = ThreadPoolExecutor(max_workers=min(args.workers, len(SUNLAKES_TRACKING_NAMES)))


    trackers = {}
    out = {"results": {}, "meta": None}

    last_scene = None

    for fr in frames:
        scene_id = fr["scene_id"]
        token = fr["token"]
        frame_idx = int(fr["frame_idx_in_scene"])

        # scene 切换 -> reset
        if last_scene != scene_id:
            trackers = {}
            ID_start_global = 1
            for cls in SUNLAKES_TRACKING_NAMES:
                trackers[cls] = SunLakesAB3DMOT(
                    cfg, cls,
                    calib=None, oxts=None, img_dir=None, vis_dir=None, hw=None, log=DummyLog(),
                    ID_init=ID_start_global
                )
                # 给下一类一个偏移，避免跨类 ID 冲突
                ID_start_global += 1000000
            last_scene = scene_id

        det_list = predictions.get(token, [])
        annos_token = []
        futs = []
        for cls in SUNLAKES_TRACKING_NAMES:
            futs.append(executor.submit(
               process_one_class,
               cls, det_list, trackers[cls],
               frame_idx, scene_id, args, token
            ))

        for fut in as_completed(futs):
            annos_token.extend(fut.result())

        out["results"][token] = annos_token

    out["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    out_path = work_dir / args.out_name
    with open(out_path, "w") as f:
        json.dump(out, f)

    c = Counter()
    for _, lst in out["results"].items():
        for p in lst:
            c[p["tracking_name"]] += 1
    print("[OK] saved:", out_path)
    print("[STATS] per-class track boxes:", dict(c))

    run_eval(args, out_path)
    executor.shutdown(wait=True)

if __name__ == "__main__":
    main()
