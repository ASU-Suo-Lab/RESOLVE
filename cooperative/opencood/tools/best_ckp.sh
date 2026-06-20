#!/usr/bin/env bash
set -euo pipefail

cd /home/suolab/OpenCOOD

OPENPCDET_DATA_ROOT_NO_FUSION="/home/suolab/OpenCOOD/data/sunlakes_opencood_ego_os128b/openpcdet_eval/v1.0-trainval"
OPENPCDET_DATA_ROOT_OS128="/home/suolab/OpenCOOD/data/sunlakes_opencood_ego_os128b_os128/openpcdet_eval/v1.0-trainval"
OPENPCDET_ROOT="/home/suolab/OpenPCDet"

METHODS=(
  "no_fusion|opencood/logs/sunlakes_point_pillar_no_fusion_2026_05_04_12_37_30|early"
  "late_fusion|opencood/logs/sunlakes_point_pillar_late_fusion_2026_05_04_16_20_54|late"
  "early_fusion|opencood/logs/sunlakes_point_pillar_early_fusion_2026_05_04_08_52_24|early"
  "attentive_fusion|opencood/logs/sunlakes_point_pillar_attentive_fusion_2026_05_04_23_08_14|intermediate"
  "cobevt|opencood/logs/sunlakes_point_pillar_cobevt_2026_05_05_12_12_14|intermediate"
  "v2xvit|opencood/logs/sunlakes_point_pillar_v2xvit_2026_05_05_14_11_20|intermediate"
)

EPOCHS=(5 10 15 20)

echo "[cleanup] stop old eval processes if any"
pkill -f "opencood/tools/eval_sunlakes.py" || true

run_eval() {
  local method="$1"
  local model_dir="$2"
  local fusion="$3"
  local epoch="$4"
  local save_dir="$model_dir/sunlakes_eval_epoch${epoch}"
  local log_file="/tmp/${method}_epoch${epoch}.log"
  local data_root="$OPENPCDET_DATA_ROOT_OS128"
  local pids=()
  local shard_logs=()

  if [[ "$method" == "no_fusion" ]]; then
    data_root="$OPENPCDET_DATA_ROOT_NO_FUSION"
  fi

  echo "[$(date +%F' '%T)] START method=$method epoch=$epoch save_dir=$save_dir" | tee "$log_file"

  rm -f "$save_dir"/results_nusc_shard*_of_4.json "$save_dir"/results_nusc.json

  for gpu in 0 1 2 3; do
    local shard_log="/tmp/${method}_epoch${epoch}_shard${gpu}.log"
    shard_logs+=("$shard_log")
    : > "$shard_log"
    CUDA_VISIBLE_DEVICES="$gpu" python opencood/tools/eval_sunlakes.py \
      --model_dir "$model_dir" \
      --checkpoint "net_epoch${epoch}.pth" \
      --fusion_method "$fusion" \
      --openpcdet_data_root "$data_root" \
      --openpcdet_root "$OPENPCDET_ROOT" \
      --num_workers 0 \
      --save_dir "$save_dir" \
      --shard_id "$gpu" \
      --num_shards 4 \
      > "$shard_log" 2>&1 &
    pids+=("$!")
  done

  local finished=0
  while (( finished == 0 )); do
    finished=1
    local summary="[$(date +%F' '%T)] PROGRESS method=$method epoch=$epoch"
    for i in 0 1 2 3; do
      local pid="${pids[$i]}"
      local shard_status="done"
      if kill -0 "$pid" 2>/dev/null; then
        finished=0
        shard_status="starting"
      fi
      local progress_line
      progress_line="$(python - <<'PY' "${shard_logs[$i]}"
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(errors='ignore') if path.exists() else ''
matches = re.findall(r'Running inference shard (\d+)/(\d+):\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)', text, flags=re.S)
if matches:
    shard_id, shard_total, pct, cur, total = matches[-1]
    print(f"shard{shard_id}={cur}/{total}({pct}%)")
elif 'Finished shard' in text:
    print('done')
elif 'loaded checkpoint' in text:
    print('loaded')
else:
    print('starting')
PY
)"
      if [[ "$progress_line" == "done" ]]; then
        shard_status="done"
      elif [[ "$progress_line" != "starting" && "$progress_line" != "loaded" ]]; then
        shard_status="$progress_line"
      else
        shard_status="$progress_line"
      fi
      summary+=" | ${shard_status}"
    done
    echo "$summary"
    if (( finished == 0 )); then
      sleep 30
    fi
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  cat "${shard_logs[0]}" "${shard_logs[1]}" "${shard_logs[2]}" "${shard_logs[3]}" > "$log_file"

  echo "[$(date +%F' '%T)] MERGE method=$method epoch=$epoch save_dir=$save_dir"

  CUDA_VISIBLE_DEVICES="" python opencood/tools/eval_sunlakes.py \
    --model_dir "$model_dir" \
    --checkpoint "net_epoch${epoch}.pth" \
    --fusion_method "$fusion" \
    --openpcdet_data_root "$data_root" \
    --openpcdet_root "$OPENPCDET_ROOT" \
    --save_dir "$save_dir" \
    --num_shards 4 \
    --aggregate_only \
    >> "$log_file" 2>&1

  echo "[$(date +%F' '%T)] DONE method=$method epoch=$epoch log=$log_file"
}

echo "[eval] start epoch sweep"
for entry in "${METHODS[@]}"; do
  IFS="|" read -r method model_dir fusion <<< "$entry"
  for epoch in "${EPOCHS[@]}"; do
    run_eval "$method" "$model_dir" "$fusion" "$epoch"
  done
done

echo "[summary] collecting best checkpoints"
python - <<'PY'
import json
from pathlib import Path

methods = {
    "no_fusion": "opencood/logs/sunlakes_point_pillar_no_fusion_2026_05_04_12_37_30",
    "late_fusion": "opencood/logs/sunlakes_point_pillar_late_fusion_2026_05_04_16_20_54",
    "early_fusion": "opencood/logs/sunlakes_point_pillar_early_fusion_2026_05_04_08_52_24",
    "attentive_fusion": "opencood/logs/sunlakes_point_pillar_attentive_fusion_2026_05_04_23_08_14",
    "cobevt": "opencood/logs/sunlakes_point_pillar_cobevt_2026_05_05_12_12_14",
    "v2xvit": "opencood/logs/sunlakes_point_pillar_v2xvit_2026_05_05_14_11_20",
}
epochs = [5, 10, 15, 20]

rows = []
for method, model_dir in methods.items():
    best = None
    all_scores = []
    for epoch in epochs:
        p = Path(model_dir) / f"sunlakes_eval_epoch{epoch}" / "metrics_summary.json"
        if not p.exists():
            continue
        data = json.load(open(p))
        m_ap = data["mean_ap"]
        nds = data["nd_score"]
        all_scores.append((epoch, m_ap, nds))
        if best is None or m_ap > best[1]:
            best = (epoch, m_ap, nds)

    print(f"\n{method}:")
    for epoch, m_ap, nds in all_scores:
        print(f"  epoch{epoch}: mAP={m_ap:.4f} NDS={nds:.4f}")
    if best:
        rows.append((method, best[0], best[1], best[2]))
        print(f"  BEST: epoch{best[0]} mAP={best[1]:.4f} NDS={best[2]:.4f}")

print("\n== Ranking by best mAP ==")
for rank, (method, epoch, m_ap, nds) in enumerate(sorted(rows, key=lambda x: x[2], reverse=True), 1):
    print(f"{rank}. {method:18s} epoch{epoch:<2d} mAP={m_ap:.4f} NDS={nds:.4f}")
PY
