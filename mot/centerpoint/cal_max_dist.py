#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compute_gating_thresholds.py

统计 SunLakes 数据集中“按类别的最大允许匹配距离（gating阈值）”
统计定义（与 CenterTrack 匹配对齐）：
  err = || (cur_xy - cur_vxy * dt) - prev_xy ||_2

输入：
  <data_root>/sunlakes_infos_train.pkl
  <data_root>/sunlakes_infos_val.pkl (可选)

输出：
  - 控制台打印每类样本数、dt分布、误差分位数、建议阈值
  - 可选保存 json

用法示例：
  python compute_gating_thresholds.py --data_root /scratch/mgarci84/sunlakes --split train --pctl 99.9
  python compute_gating_thresholds.py --data_root /scratch/mgarci84/sunlakes --split all --save_json gating.json
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_infos(data_root: Path, split: str):
    infos = []
    if split in ("train", "all"):
        p = data_root / "sunlakes_infos_train.pkl"
        if p.exists():
            with open(p, "rb") as f:
                infos.extend(pickle.load(f))
        else:
            print(f"[WARN] not found: {p}")
    if split in ("val", "all"):
        p = data_root / "sunlakes_infos_val.pkl"
        if p.exists():
            with open(p, "rb") as f:
                infos.extend(pickle.load(f))
        else:
            print(f"[WARN] not found: {p}")
    return infos


def safe_float_ts(info):
    """
    你当前生成逻辑里 timestamp 是 sec + nano/1e9 的 float
    """
    ts = info.get("timestamp", None)
    if ts is None:
        return None
    try:
        return float(ts)
    except Exception:
        return None


def compute_thresholds(
    infos,
    time_lag_target=0.5,
    percentile=99.9,
    min_dt=1e-6,
    max_dt=2.0,
    normalize_to_target=True,
    require_velocity=True,
):
    """
    normalize_to_target=True:
      如果帧间 dt != time_lag_target，把误差按 (time_lag_target/dt) 线性缩放到目标 time_lag 上，
      方便你直接拿去当 time_lag_target 条件下的 gating（CenterTrack 常用 0.5s）。

    require_velocity=True:
      要求 gt_boxes 中必须有 vx,vy（第7~8列），否则跳过。
    """

    # 先按 scene_id 分组，再按 timestamp 排序
    by_scene = defaultdict(list)
    for info in infos:
        scene_id = info.get("scene_id", None)
        ts = safe_float_ts(info)
        if scene_id is None or ts is None:
            continue
        by_scene[scene_id].append((ts, info))

    # 统计容器
    errs_by_cls = defaultdict(list)
    dts_by_cls = defaultdict(list)
    pairs_by_cls = defaultdict(int)
    skipped_no_vel = 0
    skipped_bad_dt = 0
    skipped_missing = 0

    # 每个 scene 内：把同一个 tracking_id 的轨迹串起来
    for scene_id, items in by_scene.items():
        items.sort(key=lambda x: x[0])  # sort by ts
        # 建立 track_id -> list[(ts, idx_in_frame, cls, xy, vxy)]
        tracks = defaultdict(list)

        for ts, info in items:
            gt_boxes = info.get("gt_boxes", None)
            gt_names = info.get("gt_names", None)
            tracking_ids = info.get("tracking_id", None)

            if gt_boxes is None or gt_names is None or tracking_ids is None:
                skipped_missing += 1
                continue

            gt_boxes = np.asarray(gt_boxes)
            gt_names = np.asarray(gt_names)

            # 期望 gt_boxes: (N, 9) [x,y,z, dx,dy,dz, yaw, vx, vy]
            if require_velocity and (gt_boxes.shape[1] < 9):
                skipped_no_vel += 1
                continue

            # tracking_id 可能是 list[str]
            if len(tracking_ids) != gt_boxes.shape[0]:
                skipped_missing += 1
                continue

            xy = gt_boxes[:, 0:2].astype(np.float32)
            if gt_boxes.shape[1] >= 9:
                vxy = gt_boxes[:, 7:9].astype(np.float32)
            else:
                vxy = np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)

            for i, tid in enumerate(tracking_ids):
                cls = str(gt_names[i])
                tracks[tid].append((ts, cls, xy[i], vxy[i]))

        # 对每条轨迹计算相邻帧误差
        for tid, seq in tracks.items():
            if len(seq) < 2:
                continue
            seq.sort(key=lambda x: x[0])
            for k in range(1, len(seq)):
                ts_prev, cls_prev, xy_prev, _ = seq[k - 1]
                ts_cur, cls_cur, xy_cur, vxy_cur = seq[k]

                # 类别以“当前帧”为准（也可改成 prev/一致性检查）
                cls = cls_cur

                dt = float(ts_cur - ts_prev)
                if not (dt > min_dt and dt < max_dt):
                    skipped_bad_dt += 1
                    continue

                # CenterTrack式回推：cur_xy - cur_v * dt 应该接近 prev_xy
                pred_prev = xy_cur - vxy_cur * dt
                err = float(np.linalg.norm(pred_prev - xy_prev))

                if normalize_to_target and time_lag_target is not None:
                    # 线性缩放到目标 time_lag（经验做法，便于直接用于固定 time_lag 的 tracker）
                    err = err * (float(time_lag_target) / dt)

                errs_by_cls[cls].append(err)
                dts_by_cls[cls].append(dt)
                pairs_by_cls[cls] += 1

    # 汇总统计
    results = {}
    for cls, errs in errs_by_cls.items():
        arr = np.array(errs, dtype=np.float32)
        dta = np.array(dts_by_cls[cls], dtype=np.float32) if len(dts_by_cls[cls]) else None

        if arr.size == 0:
            continue

        # 常用分位数
        pct_list = [50, 90, 95, 99, 99.5, 99.9]
        pct_vals = {str(p): float(np.percentile(arr, p)) for p in pct_list}
        gate = float(np.percentile(arr, percentile))

        res = {
            "count_pairs": int(pairs_by_cls[cls]),
            "err_mean": float(arr.mean()),
            "err_std": float(arr.std()),
            "err_min": float(arr.min()),
            "err_max": float(arr.max()),
            "err_percentiles_m": pct_vals,
            "suggest_gating_m": gate,
        }

        if dta is not None and dta.size > 0:
            res.update({
                "dt_mean": float(dta.mean()),
                "dt_min": float(dta.min()),
                "dt_max": float(dta.max()),
            })

        results[cls] = res

    meta = {
        "time_lag_target": time_lag_target,
        "normalize_to_target": normalize_to_target,
        "percentile_used": percentile,
        "min_dt": min_dt,
        "max_dt": max_dt,
        "skipped_no_vel": skipped_no_vel,
        "skipped_bad_dt": skipped_bad_dt,
        "skipped_missing": skipped_missing,
    }

    return results, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,default='/scratch/mgarci84/sunlakes/high_2lidar_cam/v1.0-trainval/', help="sunlakes 数据根目录")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "all"])
    parser.add_argument("--time_lag", type=float, default=0.5, help="目标 time_lag（用于可选归一化）")
    parser.add_argument("--pctl", type=float, default=99, help="建议 gating 使用的分位数")
    parser.add_argument("--min_dt", type=float, default=1e-6)
    parser.add_argument("--max_dt", type=float, default=2.0)
    parser.add_argument("--no_normalize", action="store_true", help="不把误差缩放到 time_lag 上")
    parser.add_argument("--allow_no_vel", action="store_true", help="允许 gt_boxes 没有 vx,vy（则用 0）")
    parser.add_argument("--save_json", type=str, default=None, help="保存结果到 json 文件（相对/绝对路径都行）")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    infos = load_infos(data_root, args.split)
    print(f"[INFO] loaded infos: {len(infos)} (split={args.split})")

    results, meta = compute_thresholds(
        infos,
        time_lag_target=args.time_lag,
        percentile=args.pctl,
        min_dt=args.min_dt,
        max_dt=args.max_dt,
        normalize_to_target=(not args.no_normalize),
        require_velocity=(not args.allow_no_vel),
    )

    # 按类别名排序打印
    print("\n========== META ==========")
    print(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\n========== PER-CLASS STATS ==========")
    for cls in sorted(results.keys()):
        r = results[cls]
        print(f"\n[{cls}] pairs={r['count_pairs']}")
        if "dt_mean" in r:
            print(f"  dt: mean={r['dt_mean']:.4f}, min={r['dt_min']:.4f}, max={r['dt_max']:.4f}")
        print(f"  err: mean={r['err_mean']:.4f}, std={r['err_std']:.4f}, min={r['err_min']:.4f}, max={r['err_max']:.4f}")
        pcts = r["err_percentiles_m"]
        print("  percentiles(m): " + ", ".join([f"p{p}={pcts[p]:.4f}" for p in ["50","90","95","99","99.5","99.9"]]))
        print(f"  ==> suggested_gating_m (p{args.pctl}) = {r['suggest_gating_m']:.4f}")

    # 生成可直接复制进代码的 dict
    gating_dict = {cls: float(results[cls]["suggest_gating_m"]) for cls in results.keys()}
    print("\n========== COPY-PASTE DICT ==========")
    print("SUNLAKES_CLS_GATING = {")
    for cls in sorted(gating_dict.keys()):
        print(f"  '{cls}': {gating_dict[cls]:.6g},")
    print("}")

    if args.save_json:
        out_path = Path(args.save_json)
        if not out_path.is_absolute():
            out_path = data_root / out_path
        payload = {"meta": meta, "per_class": results, "suggest_dict": gating_dict}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
