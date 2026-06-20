#!/usr/bin/env bash
set -euo pipefail

OPENCOOD_ROOT="/home/suolab/OpenCOOD"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
EVAL_ROOT="${EVAL_ROOT:-/home/suolab/OpenCOOD/data/sunlakes_opencood_ego_os128b/openpcdet_eval/v1.0-trainval}"
OPENPCDET_ROOT="${OPENPCDET_ROOT:-/home/suolab/OpenPCDet}"
LOG_ROOT="${LOG_ROOT:-/home/suolab/OpenCOOD/opencood/logs}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_SAVE_PER_GPU="${EVAL_SAVE_PER_GPU:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_sunlakes_4gpu.sh train <method>
  run_sunlakes_4gpu.sh eval <method> <model_dir> [gpu_id]
  run_sunlakes_4gpu.sh eval_latest <method> [gpu_id]
  run_sunlakes_4gpu.sh eval4_latest <method>
  run_sunlakes_4gpu.sh eval4_multi_latest <method1> <method2> [method3] [method4]
  run_sunlakes_4gpu.sh latest_dir <method>

Methods:
  no_fusion | late_fusion | early_fusion | attentive_fusion | fcooper | cobevt | v2xvit

Environment overrides:
  PYTHON_BIN, TRAIN_GPUS, NPROC_PER_NODE, MASTER_PORT, EVAL_ROOT, OPENPCDET_ROOT, LOG_ROOT
  EVAL_NUM_WORKERS, EVAL_MAX_SAMPLES, EVAL_SAVE_PER_GPU
EOF
}

method_yaml() {
  case "$1" in
    no_fusion) echo "$OPENCOOD_ROOT/opencood/hypes_yaml/sunlakes_point_pillar_no_fusion.yaml" ;;
    late_fusion) echo "$OPENCOOD_ROOT/opencood/hypes_yaml/sunlakes_point_pillar_late_fusion.yaml" ;;
    early_fusion) echo "$OPENCOOD_ROOT/opencood/hypes_yaml/sunlakes_point_pillar_early_fusion.yaml" ;;
    attentive_fusion) echo "$OPENCOOD_ROOT/opencood/hypes_yaml/sunlakes_point_pillar_attentive_fusion.yaml" ;;
    fcooper) echo "$OPENCOOD_ROOT/opencood/hypes_yaml/sunlakes_point_pillar_fcooper.yaml" ;;
    cobevt) echo "$OPENCOOD_ROOT/opencood/hypes_yaml/sunlakes_point_pillar_cobevt.yaml" ;;
    v2xvit) echo "$OPENCOOD_ROOT/opencood/hypes_yaml/sunlakes_point_pillar_v2xvit.yaml" ;;
    *) echo "Unknown method: $1" >&2; exit 1 ;;
  esac
}

method_name() {
  case "$1" in
    no_fusion) echo "sunlakes_point_pillar_no_fusion" ;;
    late_fusion) echo "sunlakes_point_pillar_late_fusion" ;;
    early_fusion) echo "sunlakes_point_pillar_early_fusion" ;;
    attentive_fusion) echo "sunlakes_point_pillar_attentive_fusion" ;;
    fcooper) echo "sunlakes_point_pillar_fcooper" ;;
    cobevt) echo "sunlakes_point_pillar_cobevt" ;;
    v2xvit) echo "sunlakes_point_pillar_v2xvit" ;;
    *) echo "Unknown method: $1" >&2; exit 1 ;;
  esac
}

method_fusion() {
  case "$1" in
    no_fusion) echo "early" ;;
    late_fusion) echo "late" ;;
    early_fusion) echo "early" ;;
    attentive_fusion|fcooper|cobevt|v2xvit) echo "intermediate" ;;
    *) echo "Unknown method: $1" >&2; exit 1 ;;
  esac
}

latest_dir() {
  local method="$1"
  local prefix
  prefix="$(method_name "$method")"
  local found
  found="$(ls -dt "$LOG_ROOT/${prefix}"_* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$found" ]]; then
    echo "No log directory found for method=$method under $LOG_ROOT" >&2
    exit 1
  fi
  echo "$found"
}

train_method() {
  local method="$1"
  local yaml
  yaml="$(method_yaml "$method")"
  echo "[train] method=$method"
  echo "[train] yaml=$yaml"
  echo "[train] gpus=$TRAIN_GPUS nproc=$NPROC_PER_NODE master_port=$MASTER_PORT"
  cd "$OPENCOOD_ROOT"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"   MASTER_PORT="$MASTER_PORT"   "$PYTHON_BIN" -m torch.distributed.launch     --nproc_per_node="$NPROC_PER_NODE"     --use_env     opencood/tools/train.py     --hypes_yaml "$yaml"
}

eval_method() {
  local method="$1"
  local model_dir="$2"
  local gpu_id="${3:-0}"
  local fusion
  local save_dir=""
  local extra_args=()
  fusion="$(method_fusion "$method")"
  if [[ "$EVAL_SAVE_PER_GPU" == "1" ]]; then
    save_dir="$model_dir/sunlakes_eval_gpu${gpu_id}"
    extra_args+=(--save_dir "$save_dir")
  fi
  echo "[eval] method=$method"
  echo "[eval] model_dir=$model_dir"
  echo "[eval] gpu=$gpu_id fusion=$fusion workers=$EVAL_NUM_WORKERS max_samples=$EVAL_MAX_SAMPLES save_dir=${save_dir:-<default>}"
  cd "$OPENCOOD_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    "$PYTHON_BIN" opencood/tools/eval_sunlakes.py \
    --model_dir "$model_dir" \
    --fusion_method "$fusion" \
    --openpcdet_data_root "$EVAL_ROOT" \
    --openpcdet_root "$OPENPCDET_ROOT" \
    --num_workers "$EVAL_NUM_WORKERS" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    "${extra_args[@]}"
}

eval_shard_method() {
  local method="$1"
  local model_dir="$2"
  local gpu_id="$3"
  local shard_id="$4"
  local num_shards="$5"
  local fusion
  local save_dir="$model_dir/sunlakes_eval_4gpu"
  fusion="$(method_fusion "$method")"

  echo "[eval-shard] method=$method model_dir=$model_dir gpu=$gpu_id shard=$shard_id/$num_shards workers=$EVAL_NUM_WORKERS max_samples=$EVAL_MAX_SAMPLES save_dir=$save_dir"
  cd "$OPENCOOD_ROOT"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    "$PYTHON_BIN" opencood/tools/eval_sunlakes.py \
    --model_dir "$model_dir" \
    --fusion_method "$fusion" \
    --openpcdet_data_root "$EVAL_ROOT" \
    --openpcdet_root "$OPENPCDET_ROOT" \
    --num_workers "$EVAL_NUM_WORKERS" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --save_dir "$save_dir" \
    --shard_id "$shard_id" \
    --num_shards "$num_shards"
}

aggregate_shards() {
  local method="$1"
  local model_dir="$2"
  local num_shards="$3"
  local fusion
  local save_dir="$model_dir/sunlakes_eval_4gpu"
  fusion="$(method_fusion "$method")"

  echo "[eval-merge] method=$method model_dir=$model_dir shards=$num_shards save_dir=$save_dir"
  cd "$OPENCOOD_ROOT"
  CUDA_VISIBLE_DEVICES="" \
    "$PYTHON_BIN" opencood/tools/eval_sunlakes.py \
    --model_dir "$model_dir" \
    --fusion_method "$fusion" \
    --openpcdet_data_root "$EVAL_ROOT" \
    --openpcdet_root "$OPENPCDET_ROOT" \
    --save_dir "$save_dir" \
    --num_shards "$num_shards" \
    --aggregate_only
}

eval_latest() {
  local method="$1"
  local gpu_id="${2:-0}"
  local model_dir
  model_dir="$(latest_dir "$method")"
  eval_method "$method" "$model_dir" "$gpu_id"
}

eval4_latest() {
  if [[ $# -ne 1 ]]; then
    echo "eval4_latest expects exactly 1 method" >&2
    exit 1
  fi
  local method="$1"
  local model_dir
  model_dir="$(latest_dir "$method")"
  local gpus=(0 1 2 3)
  local pids=()
  local idx=0
  for gpu in "${gpus[@]}"; do
    local shard_id="$idx"
    echo "[eval4] launching method=$method shard=$shard_id/4 on gpu=$gpu"
    (
      eval_shard_method "$method" "$model_dir" "$gpu" "$shard_id" 4
    ) &
    pids+=("$!")
    idx=$((idx + 1))
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
  done
  if [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi
  aggregate_shards "$method" "$model_dir" 4
}

eval4_multi_latest() {
  if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "eval4_multi_latest expects 2 to 4 methods" >&2
    exit 1
  fi
  local gpus=(0 1 2 3)
  local pids=()
  local idx=0
  for method in "$@"; do
    local gpu="${gpus[$idx]}"
    echo "[eval4-multi] launching method=$method on gpu=$gpu"
    (
      eval_latest "$method" "$gpu"
    ) &
    pids+=("$!")
    idx=$((idx + 1))
  done
  local rc=0
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=$?
  done
  return "$rc"
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd="$1"
  shift
  case "$cmd" in
    train)
      [[ $# -eq 1 ]] || { usage; exit 1; }
      train_method "$1"
      ;;
    eval)
      [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 1; }
      eval_method "$@"
      ;;
    eval_latest)
      [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 1; }
      eval_latest "$@"
      ;;
    eval4_latest)
      eval4_latest "$@"
      ;;
    eval4_multi_latest)
      eval4_multi_latest "$@"
      ;;
    latest_dir)
      [[ $# -eq 1 ]] || { usage; exit 1; }
      latest_dir "$1"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
