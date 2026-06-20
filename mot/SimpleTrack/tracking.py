import os
import json
import yaml
import argparse
import pickle
import numpy as np
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from mot_3d.data_protos import BBox
from mot_3d.mot import MOTModel
from mot_3d.frame_data import FrameData

from scipy.optimize import linear_sum_assignment

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to sunlakes_infos_train.pkl or sunlakes_infos_val.pkl')
    parser.add_argument("--split", type=str, default="trainval", choices=["trainval","val"])
    parser.add_argument('--pred_json', type=str, required=True,
                        help='Input detection json (token->detections)')
    parser.add_argument('--config_path', type=str, default='configs/nu_configs/giou.yaml')
    parser.add_argument('--work_dir', type=str, default="./results")
    parser.add_argument('--obj_types', default='car,bus,trailer,truck,pedestrian,bicycle,motorcycle,construction_vehicle,barrier,traffic_cone')
    # If split_by_scene=True, we reset tracker per scene to avoid id contamination
    parser.add_argument('--split_by_scene', action='store_true', default=True)

    parser.add_argument('--workers', type=int, default=16, help='threads for parallel tracking/eval')

    # ===== Eval args =====
    parser.add_argument('--dist_th', type=float, default=2.0, help='distance threshold for matching (meters)')
    parser.add_argument('--num_thresholds', type=int, default=40, help='recall levels for AMOTA curve')
    parser.add_argument('--min_recall', type=float, default=0.1, help='min recall level for AMOTA averaging')
    parser.add_argument('--max_switch_time', type=int, default=5, help='max frames to count an ID switch')
    parser.add_argument('--mota_score_th', type=float, default=0.5, help='fixed score threshold for MOTA/MOTP point')

    return parser.parse_args()


# -------------------------
# Utils: yaw <-> quaternion (z-axis)
# -------------------------
def yaw_to_quat_wxyz(yaw: float):
    half = yaw * 0.5
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]

def quat_wxyz_to_yaw(q):
    qw, qx, qy, qz = q
    return float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))


def track_one_class_one_seq(obj_type, seq_infos, seq_idx, det_results, configs, split_by_scene):
    """
    Run MOTModel for one class on one sequence (scene).
    Returns: list of (token, anno_dict)
    """
    tracker = MOTModel(configs)
    outputs = []

    for frame_idx, info in enumerate(seq_infos):
        token = info['token']
        ts = info.get('timestamp', None)
        time_stamp = float(ts) if ts is not None else float(frame_idx)

        annos = det_results.get(token, [])
        dets, det_types = [], []

        for a in annos:
            if a.get('detection_name') != obj_type:
                continue
            dets.append(det_anno_to_mot_array(a))
            det_types.append(obj_type)

        fd = FrameData(
            dets=dets,
            ego=None,
            pc=None,
            det_types=det_types,
            aux_info={'is_key_frame': True, 'token': token},
            time_stamp=time_stamp
        )

        results = tracker.frame_mot(fd)

        for trk_bbox, trk_id, trk_state, trk_type in results:
            out_id = trk_id
            if split_by_scene:
                out_id = f"s{seq_idx}_{trk_id}"

            outputs.append((token, mot_bbox_to_track_anno(trk_bbox, out_id, obj_type, token)))

    return outputs


# -------------------------
# Convert det anno -> mot bbox array
# size order confirmed: wlh
# -------------------------
def det_anno_to_mot_array(a):
    xyz = a['translation']
    wlh = a['size']  # [w,l,h]
    yaw = quat_wxyz_to_yaw(a['rotation'])
    score = float(a.get('detection_score', 1.0))

    w, l, h = float(wlh[0]), float(wlh[1]), float(wlh[2])

    # FrameData -> BBox.array2bbox expects: x,y,z,o,l,w,h,(score)
    arr = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]),
                    float(yaw), l, w, h, score], dtype=np.float32)
    return arr


def build_sequence_from_infos(infos, split_by_scene: bool, sort_by_timestamp: bool = True):
    """
    infos: list[dict], each has token, timestamp, scene_id
    returns: list of sequences, each sequence is a list[info] in time order
    """
    has_scene = any(('scene_id' in x) for x in infos)

    if split_by_scene and has_scene:
        buckets = defaultdict(list)
        for info in infos:
            buckets[info.get('scene_id', 'default_scene')].append(info)

        sequences = []
        for sid in sorted(buckets.keys()):
            seq = buckets[sid]
            if sort_by_timestamp and all(('timestamp' in x) for x in seq):
                seq = sorted(seq, key=lambda x: float(x['timestamp']))
            else:
                seq = sorted(seq, key=lambda x: str(x['token']))
            sequences.append(seq)
        return sequences

    # single sequence
    seq = list(infos)
    if sort_by_timestamp and all(('timestamp' in x) for x in seq):
        seq = sorted(seq, key=lambda x: float(x['timestamp']))
    else:
        seq = sorted(seq, key=lambda x: str(x['token']))
    return [seq]


# -------------------------
# Convert mot bbox -> track anno json
# -------------------------
def mot_bbox_to_track_anno(bbox: BBox, track_id: str, track_name: str, sample_token: str):
    yaw = float(bbox.o)
    q = yaw_to_quat_wxyz(yaw)
    return {
        "sample_token": sample_token,
        "translation": [float(bbox.x), float(bbox.y), float(bbox.z)],
        "size": [float(bbox.w), float(bbox.l), float(bbox.h)],  # wlh
        "rotation": q,                                          # [w,x,y,z]
        "velocity": [0.0, 0.0],                                 # placeholder
        "tracking_name": track_name,
        "tracking_id": str(track_id),
        "tracking_score": float(getattr(bbox, 's', 1.0)),
    }


# -------------------------
# Load infos pkl: frame order + timestamp (+ scene_id)
# -------------------------
def load_infos(infos_pkl):
    with open(infos_pkl, 'rb') as f:
        infos = pickle.load(f)
    if not isinstance(infos, (list, tuple)) or len(infos) == 0:
        raise ValueError("infos_pkl must be a non-empty list[dict].")
    if 'token' not in infos[0]:
        raise KeyError("infos entries must contain 'token'.")
    if 'timestamp' not in infos[0]:
        print("[WARN] infos has no 'timestamp'. Will use frame index as time_stamp.")
    return infos


# ============================================================
# Eval: AMOTA/AMOTP + MOTA/MOTP
# ============================================================
def load_gt_pkl_from_infos(infos_pkl: Path):
    with open(infos_pkl, "rb") as f:
        infos = pickle.load(f)

    by_scene = defaultdict(list)
    for info in infos:
        scene = info.get("scene_id", "scene0")
        ts = float(info.get("timestamp", 0.0))
        token = info["token"]

        gt_names = np.array(info.get("gt_names", []), dtype=object)
        gt_boxes = np.array(info.get("gt_boxes", []), dtype=np.float32)
        gt_ids = list(info.get("tracking_id", []))

        # ensure alignment
        if gt_boxes.ndim == 1:
            gt_boxes = gt_boxes.reshape(0, 7)
        if len(gt_ids) != gt_boxes.shape[0]:
            gt_ids = [f"{scene}_dummy_{i}" for i in range(gt_boxes.shape[0])]

        gt_xy = gt_boxes[:, :2].astype(np.float32) if gt_boxes.shape[0] > 0 else np.zeros((0, 2), dtype=np.float32)

        by_scene[scene].append({
            "token": token,
            "timestamp": ts,
            "gt_names": gt_names,
            "gt_xy": gt_xy,
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


def compute_mota_motp_at_threshold(by_scene_gt, preds_by_token, cls_name, dist_th, score_th, max_switch_time):
    st = eval_one_class(by_scene_gt, preds_by_token, cls_name, dist_th, score_th, max_switch_time)
    P = st["P"]
    if P == 0:
        return {"mota": 0.0, "motp": dist_th, "note": "No GT for this class."}

    mota = 1.0 - (st["FN"] + st["FP"] + st["IDS"]) / float(P)
    motp = (st["dist_sum"] / st["match_cnt"]) if st["match_cnt"] > 0 else dist_th
    return {"mota": float(mota), "motp": float(motp), **st}


def run_eval(args, infos_pkl_path: Path, tracking_json_path: Path, class_names, workers=10):
    by_scene_gt = load_gt_pkl_from_infos(infos_pkl_path)
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

     # --- parallel per-class eval ---
    def _eval_one_cls(cls):
        am_res = compute_amota_amotp(
            by_scene_gt, preds_by_token, cls,
            dist_th=args.dist_th,
            num_thresholds=args.num_thresholds,
            min_recall=args.min_recall,
            max_switch_time=args.max_switch_time
        )
        m_res = compute_mota_motp_at_threshold(
            by_scene_gt, preds_by_token, cls,
            dist_th=args.dist_th,
            score_th=args.mota_score_th,
            max_switch_time=args.max_switch_time
        )
        return cls, am_res, m_res

    max_workers = min(int(workers), len(class_names), int(os.cpu_count() or workers))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_eval_one_cls, cls) for cls in class_names]

        for fut in tqdm(as_completed(futs), total=len(futs), desc="[EVAL] classes", unit="class"):
            cls, am_res, m_res = fut.result()

            per_class[cls] = {"amota_amotp": am_res, "mota_motp": m_res}

            amota_all.append(am_res["amota"])
            amotp_all.append(am_res["amotp"])
            mota_all.append(m_res["mota"])
            motp_all.append(m_res["motp"])

            note = ""
            if "note" in am_res: note += f" | AM-note={am_res['note']}"
            if "note" in m_res:  note += f" | M-note={m_res['note']}"

            print(
                f"[{cls:>20}] "
                f"AMOTA={am_res['amota']:.4f} AMOTP={am_res['amotp']:.4f} || "
                f"MOTA@{args.mota_score_th:.3f}={m_res['mota']:.4f} MOTP={m_res['motp']:.4f}"
                f"{note}"
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

    print("\n========== OVERALL ==========")
    print(f"mean AMOTA = {overall['mean_amota']:.4f}")
    print(f"mean AMOTP = {overall['mean_amotp']:.4f}")
    print(f"mean MOTA@{args.mota_score_th:.3f} = {overall['mean_mota']:.4f}")
    print(f"mean MOTP           = {overall['mean_motp']:.4f}")

    out = {"per_class": per_class, "overall": overall}
    out_path = Path(args.work_dir) / "amota_amotp_mota_motp.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] saved to: {out_path}")
    return out_path


# -------------------------
# Main: tracking -> results_track.json -> eval metrics json
# -------------------------
def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    # configs
    configs = yaml.load(open(args.config_path, 'r'), Loader=yaml.Loader)

    # infos: defines order + timestamp
    data_root = Path(args.data_root).expanduser()
    if data_root.is_dir():
        infos_pkl = data_root / f"sunlakes_infos_{args.split}.pkl"
    else:
        infos_pkl = data_root  # 兼容：直接传 pkl

    if not infos_pkl.exists():
        raise FileNotFoundError(f"infos pkl not found: {infos_pkl}")

    infos = load_infos(str(infos_pkl))
    sequences = build_sequence_from_infos(infos, args.split_by_scene, sort_by_timestamp=True)

    # detections
    with open(args.pred_json, 'r') as f:
        det_data = json.load(f)
    det_results = det_data['results']  # token -> det list
    meta = det_data.get('meta', None)

    obj_types = [x.strip() for x in args.obj_types.split(',') if x.strip()]

    token2tracks = defaultdict(list)

    total_frames = len(obj_types) * sum(len(seq) for seq in sequences)
    max_workers = min(args.workers, len(obj_types) * len(sequences), os.cpu_count() or args.workers)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2len = {}
        futs = []

        for obj_type in obj_types:
            for seq_idx, seq_infos in enumerate(sequences):
                fut = ex.submit(
                    track_one_class_one_seq,
                    obj_type, seq_infos, seq_idx, det_results, configs, args.split_by_scene
                )
                futs.append(fut)
                fut2len[fut] = len(seq_infos)

        pbar = tqdm(total=total_frames, desc="[Tracking] frames", unit="frame")

        for fut in as_completed(futs):
            seq_len = fut2len[fut]
            outputs = fut.result()  # list[(token, anno)]

            # merge
            for token, anno in outputs:
                token2tracks[token].append(anno)

            # progress update
            pbar.update(seq_len)

        pbar.close()

    out = {
        "results": dict(token2tracks),
        "meta": meta if meta is not None else {
            "use_camera": False, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False
        }
    }

    out_path = os.path.join(args.work_dir, 'results_track.json')
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print("\nSaved tracking results to:", out_path)

    # ---- eval ----
    class_names = [x.strip() for x in args.obj_types.split(',') if x.strip()]
    run_eval(args, infos_pkl, Path(out_path), class_names=class_names, workers=args.workers)


if __name__ == "__main__":
    main()
