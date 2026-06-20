import json
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.optimize import linear_sum_assignment
import pub_tracker 
from pub_tracker import PubTracker


SUNLAKES_TRACKING_NAMES = [
    'barrier',
    #'bicycle',
    #'bus',
    #
     'car',
    #'construction_vehicle',
    #'motorcycle',
    #'pedestrian',
    #'traffic_cone',
    #'trailer',
    #'truck'
]

# 你统计出来的 gating（示例：p99）
SUNLAKES_CLS_GATING = {
  'barrier': 2,
  'bicycle': 1.5,
  'bus': 2,
  'car': 2,
  'construction_vehicle': 1.5,
  'motorcycle': 2.5,
  'pedestrian': 1,
  'traffic_cone': 1.5,
  'trailer': 1.5,
  'truck': 2,
}


def parse_args():
    p = argparse.ArgumentParser("SunLakes: run tracking then eval AMOTA/AMOTP using pkl GT tracking_id")
    # tracking inputs
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--pred_json", type=str, required=True,
                   help="detection prediction json (results_nusc.json-like): {'results':{token:[...]}, 'meta':...}")
    p.add_argument("--work_dir", type=str, default="./results")
    p.add_argument("--hungarian", action="store_true")
    p.add_argument("--max_age", type=int, default=3)
    p.add_argument("--workers", type=int, default=10, help="threads for tracking/eval")

    # eval params
    p.add_argument("--dist_th", type=float, default=2.0)
    p.add_argument("--num_thresholds", type=int, default=40)
    p.add_argument("--min_recall", type=float, default=0.1)
    p.add_argument("--max_switch_time", type=int, default=999999)
    p.add_argument("--mota_score_th", type=float, default=0.5,
               help="固定阈值下计算 MOTA/MOTP 的 score_th（部署点）")

    return p.parse_args()


# -------------------------
# tracking: frames + normalize
# -------------------------

def compute_mota_motp_at_threshold(by_scene_gt, preds_by_token, cls_name,
                                   dist_th, score_th, max_switch_time):
    """
    在固定 score_th 下，计算该类别的 MOTA / MOTP（单点指标）。
    """
    st = eval_one_class(by_scene_gt, preds_by_token, cls_name,
                        dist_th=dist_th, score_th=score_th, max_switch_time=max_switch_time)
    P = st["P"]
    if P == 0:
        return {"mota": 0.0, "motp": dist_th, "note": "No GT for this class."}

    mota = 1.0 - (st["FN"] + st["FP"] + st["IDS"]) / float(P)
    mota = float(max(0.0, mota))  # 常见做法：下限裁剪到 0
    motp = float(st["dist_sum"] / st["match_cnt"]) if st["match_cnt"] > 0 else dist_th

    return {
        "mota": mota,
        "motp": motp,
        "P": int(P),
        "TP": int(st["TP"]), "FP": int(st["FP"]), "FN": int(st["FN"]), "IDS": int(st["IDS"]),
        "matches": int(st["match_cnt"]),
    }


def load_by_scene_from_pkl(data_root: Path, split: str):
    pkl_path = data_root / f"sunlakes_infos_{split}.pkl"
    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)

    by_scene = defaultdict(list)  # scene -> list[(ts, token)]
    for info in infos:
        scene = info.get("scene_id", "scene0")
        ts = float(info["timestamp"])
        token = info["token"]
        by_scene[scene].append((ts, token))

    for s in by_scene:
        by_scene[s].sort(key=lambda x: x[0])
    return by_scene

def _track_one_scene(scene_id: str, seq, predictions, args, gating):
    # 每个 scene 一个 tracker（线程安全关键点）
    tracker = PubTracker(max_age=args.max_age, hungarian=args.hungarian)
    tracker.NUSCENE_CLS_VELOCITY_ERROR = gating

    out_scene = {}
    last_ts = None
    for ts, token in seq:
        if last_ts is None:
            tracker.reset()
            last_ts = float(ts)

        time_lag = float(ts - last_ts)
        last_ts = float(ts)

        dets = predictions.get(token, [])
        dets2 = []
        for d in dets:
            d = normalize_pred_item(d)
            if d.get("detection_name") not in SUNLAKES_TRACKING_NAMES:
                continue
            dets2.append(d)

        tracks = tracker.step_centertrack(dets2, time_lag)

        annos = []
        for t in tracks:
            if t.get("active", 1) == 0:
                continue
            annos.append({
                "sample_token": token,
                "translation": t["translation"],
                "size": t["size"],
                "rotation": t["rotation"],
                "velocity": t["velocity"],
                "tracking_id": str(t["tracking_id"]),
                "tracking_name": t["detection_name"],
                "tracking_score": t["detection_score"],
            })
        out_scene[token] = annos

    return scene_id, out_scene


def normalize_pred_item(item: dict):
    """
    Ensure tracker needs:
      detection_name, detection_score, translation, velocity, size, rotation
    """
    # Some pipelines might output detection_* already (good)
    if "detection_name" not in item and "tracking_name" in item:
        item["detection_name"] = item["tracking_name"]
    if "detection_score" not in item and "tracking_score" in item:
        item["detection_score"] = item["tracking_score"]

    if "velocity" not in item or item["velocity"] is None:
        item["velocity"] = [0.0, 0.0]
    if len(item["velocity"]) >= 2:
        item["velocity"] = item["velocity"][:2]
    else:
        item["velocity"] = [0.0, 0.0]

    if "sample_token" not in item and "token" in item:
        item["sample_token"] = item["token"]

    return item


def run_tracking(args):
    data_root = Path(args.data_root)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # detections
    with open(args.pred_json, "r") as f:
        pred_all = json.load(f)
    predictions = pred_all["results"]

    # 覆盖 pub_tracker 全局类别表（保持你原逻辑）
    pub_tracker.NUSCENES_TRACKING_NAMES = SUNLAKES_TRACKING_NAMES

    # gating dict + fallback
    gating = dict(SUNLAKES_CLS_GATING)
    for k in SUNLAKES_TRACKING_NAMES:
        gating.setdefault(k, 1.0)

    # scene->sequence
    by_scene = load_by_scene_from_pkl(data_root, args.split)
    scenes = sorted(by_scene.keys())

    max_workers = min(int(getattr(args, "workers", 10)), len(scenes))
    print(f"[TRACK] scenes={len(scenes)} max_workers={max_workers}")

    out = {"results": {}, "meta": None}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(_track_one_scene, s, by_scene[s], predictions, args, gating)
            for s in scenes
        ]

        for fut in tqdm(as_completed(futs), total=len(futs), desc="[TRACK] scenes", unit="scene"):
            scene_id, scene_res = fut.result()
            out["results"].update(scene_res)

    out["meta"] = {
        "use_camera": False, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False
    }

    track_path = work_dir / "tracking_result.json"
    with open(track_path, "w") as f:
        json.dump(out, f)

    c = Counter()
    for tok, lst in out["results"].items():
        for p in lst:
            c[p["tracking_name"]] += 1
    print("[TRACK] per-class counts:", dict(c))
    print(f"[TRACK] saved: {track_path}")
    return track_path



# -------------------------
# eval: AMOTA/AMOTP
# -------------------------
def precompute_class_thresholds(preds_by_token, class_names):
    # 收集每类所有 tracking_score
    scores_by_cls = {c: [] for c in class_names}
    for plist in preds_by_token.values():
        for p in plist:
            c = p.get("tracking_name", None)
            if c in scores_by_cls:
                scores_by_cls[c].append(float(p.get("tracking_score", 0.0)))

    # unique + sort desc
    th_by_cls = {}
    for c, arr in scores_by_cls.items():
        if len(arr) == 0:
            th_by_cls[c] = None
        else:
            s = np.unique(np.asarray(arr, dtype=np.float32))
            s.sort()
            th_by_cls[c] = s[::-1]  # desc
    return th_by_cls


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

        # safety: ensure gt_ids length matches
        if len(gt_ids) != gt_boxes.shape[0]:
            # fall back: create dummy ids to avoid crash, but IDS will be meaningless
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


def compute_amota_amotp_with_thresholds(by_scene_gt, preds_by_token, cls_name,
                                        dist_th, num_thresholds, min_recall, max_switch_time,
                                        cand_thresholds):
    if cand_thresholds is None or len(cand_thresholds) == 0:
        return {"amota": 0.0, "amotp": dist_th, "note": "No predictions for this class."}

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

def run_eval(args, tracking_json_path: Path):
    by_scene_gt = load_gt_pkl(Path(args.data_root), args.split)
    preds_by_token = load_pred_tracking_json(tracking_json_path)

    class_names = list(SUNLAKES_TRACKING_NAMES)
    th_by_cls = precompute_class_thresholds(preds_by_token, class_names)

    max_workers = min(int(getattr(args, "workers", 10)), len(class_names))
    print(f"[EVAL] max_workers={max_workers}")

    def _eval_one_cls(cls):
        am_res = compute_amota_amotp_with_thresholds(
            by_scene_gt, preds_by_token, cls,
            dist_th=args.dist_th,
            num_thresholds=args.num_thresholds,
            min_recall=args.min_recall,
            max_switch_time=args.max_switch_time,
            cand_thresholds=th_by_cls[cls],
        )
        m_res = compute_mota_motp_at_threshold(
            by_scene_gt, preds_by_token, cls,
            dist_th=args.dist_th,
            score_th=args.mota_score_th,
            max_switch_time=args.max_switch_time
        )
        return cls, am_res, m_res

    per_class = {}
    amota_all, amotp_all, mota_all, motp_all = [], [], [], []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_eval_one_cls, cls) for cls in class_names]
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
                f"MOTA@{args.mota_score_th:.3f}={m_res['mota']:.4f} MOTP={m_res['motp']:.4f}{note}"
            )

    overall = {
        "mean_amota": float(np.mean(amota_all)) if amota_all else 0.0,
        "mean_amotp": float(np.mean(amotp_all)) if amotp_all else args.dist_th,
        "dist_th": args.dist_th,
        "num_thresholds": args.num_thresholds,
        "min_recall": args.min_recall,
        "max_switch_time": args.max_switch_time,
        "mean_mota": float(np.mean(mota_all)) if mota_all else 0.0,
        "mean_motp": float(np.mean(motp_all)) if motp_all else args.dist_th,
        "mota_score_th": args.mota_score_th
    }

    out = {"per_class": per_class, "overall": overall}
    out_path = Path(args.work_dir) / f"amota_amotp_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] saved to: {out_path}")
    return out_path




def main():
    args = parse_args()
    track_path = run_tracking(args)
    run_eval(args, track_path)


if __name__ == "__main__":
    main()
