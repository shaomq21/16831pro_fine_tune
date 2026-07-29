#!/usr/bin/env bash
# Fine-tune pi05_libero on libero_90 STUDY_SCENE4 book tasks (tasks_info.txt 82-85).
#
# All checkpoints / HF cache live on external disk (see tools/pi05_storage_env.sh).
#
# Usage:
#   bash tools/pi05_study_scene4_finetune.sh
#   CUDA_VISIBLE_DEVICES=3   training GPU (avoid 1,2,5,6,7 used by finetune_dual_masked)
#   SIM_CUDA_VISIBLE_DEVICES=4  similarity watcher GPU
#
# Env overrides:
#   STORAGE_ROOT, NUM_GPUS, BATCH_SIZE, STEPS, PI05_PRETRAINED, RESUME, SKIP_CONVERT
#   SIM_WATCH=1 (default)  periodic vision similarity on each new checkpoint
#   SIM_POLL_SEC=120       watcher poll interval (seconds)
#   CKPT_PRUNE=1           prune old checkpoints (see CKPT_KEEP)
#   CKPT_KEEP=2            keep newest N numeric checkpoint dirs on disk
#   WANDB_ENABLE=1         log training loss + periodic similarity to Weights & Biases
#   WANDB_PROJECT=pi05_study_scene4
#   WANDB_ENTITY=          optional wandb team/user
#   TARGET_LOSS=0.02       stop early when logged loss reaches target (runs monitor)
#   SKIP_FINAL_ANALYSIS=1  skip post-train feature extraction (for continue runs)
#   CONTINUE_WEIGHTS=1   load latest finetuned weights and keep training (no optimizer resume)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=pi05_storage_env.sh
source "${SCRIPT_DIR}/pi05_storage_env.sh"

DATA_ROOT="${STORAGE_ROOT}/datasets/modified_libero_rlds"
LEROBOT_REPO="local/libero_90_study_scene4"
OUTPUT_DIR="${PI05_FINETUNE_ROOT}"
ANALYSIS_DIR="${PI05_ANALYSIS_ROOT}"
PRETRAINED="${PI05_PRETRAINED}"

NUM_GPUS="${NUM_GPUS:-1}"
# finetune_dual_masked.sh uses 1,2,5,6,7 — keep pi05 on free GPUs 3/4 by default.
DUAL_MASKED_GPUS="${DUAL_MASKED_GPUS:-1,2,5,6,7}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export CUDA_VISIBLE_DEVICES

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-3000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
RESUME="${RESUME:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
CKPT_PRUNE="${CKPT_PRUNE:-1}"
CKPT_KEEP="${CKPT_KEEP:-2}"
SIM_WATCH="${SIM_WATCH:-1}"
SIM_POLL_SEC="${SIM_POLL_SEC:-120}"
SIM_CUDA_VISIBLE_DEVICES="${SIM_CUDA_VISIBLE_DEVICES:-4}"
WANDB_ENABLE="${WANDB_ENABLE:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-pi05_study_scene4}"
WANDB_ENTITY="${WANDB_ENTITY:-maggiesh-carnegie-mellon-university}"
TARGET_LOSS="${TARGET_LOSS:-}"
SKIP_FINAL_ANALYSIS="${SKIP_FINAL_ANALYSIS:-0}"
CONTINUE_WEIGHTS="${CONTINUE_WEIGHTS:-0}"
WANDB_RUN_ID_OVERRIDE=""
PERIODIC_ANALYSIS_DIR="${ANALYSIS_DIR}/periodic"

PYTHON="${PYTHON:-/home/fan-test/miniconda3/envs/wav_new/bin/python}"

# Migrate legacy analysis artifacts out of finetune output dir (they block lerobot cold start).
if [[ -d "${OUTPUT_DIR}/vision_features_pretrained" ]]; then
  mkdir -p "${ANALYSIS_DIR}"
  if [[ ! -d "${ANALYSIS_DIR}/vision_features_pretrained" ]]; then
    mv "${OUTPUT_DIR}/vision_features_pretrained" "${ANALYSIS_DIR}/"
    echo "Moved legacy vision_features_pretrained -> ${ANALYSIS_DIR}/"
  else
    rm -rf "${OUTPUT_DIR}/vision_features_pretrained"
  fi
fi

echo "===== Storage (external disk) ====="
pi05_storage_env

echo "===== Step 0: ensure pi05_libero + paligemma tokenizer + transformers patch ====="
bash "${SCRIPT_DIR}/download_pi05_libero.sh"
bash "${SCRIPT_DIR}/download_paligemma_tokenizer.sh"
"$PYTHON" "${SCRIPT_DIR}/patch_pi05_preprocessor_tokenizer.py" \
  --ckpt_dir "$PRETRAINED" \
  --tokenizer_dir "${PI05_PALIGEMMA_TOKENIZER}"
OPENPI_ROOT="${SCRIPT_DIR}/../third_party/openpi"
if [[ -d "${OPENPI_ROOT}/src/openpi/models_pytorch/transformers_replace" ]]; then
  TRANSFORMERS_DIR="$("$PYTHON" -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")"
  cp -r "${OPENPI_ROOT}/src/openpi/models_pytorch/transformers_replace/"* "${TRANSFORMERS_DIR}/" 2>/dev/null || true
fi

echo "===== Step 1: ensure lerobot ====="
if ! "$PYTHON" -c "import lerobot" 2>/dev/null; then
  echo "Installing lerobot..."
  "$PYTHON" -m pip install -q "lerobot>=0.3.0" "transformers==4.53.2"
fi

if [[ "$SKIP_CONVERT" != "1" ]]; then
  echo "===== Step 2: convert RLDS -> LeRobot ====="
  "$PYTHON" "${SCRIPT_DIR}/convert_study_scene4_to_lerobot.py" \
    --data_dir "$DATA_ROOT" \
    --repo_id "$LEROBOT_REPO" \
    --lerobot_home "$HF_LEROBOT_HOME"
else
  echo "===== Step 2: skip convert (SKIP_CONVERT=1) ====="
fi

if [[ "$CONTINUE_WEIGHTS" == "1" ]]; then
  SOURCE_ROOT="${CONTINUE_FROM_ROOT:-$PI05_FINETUNE_ROOT}"
  LATEST_PRETRAINED=$(ls -d "$SOURCE_ROOT"/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort -V | tail -1 || true)
  if [[ -z "$LATEST_PRETRAINED" || ! -f "$LATEST_PRETRAINED/model.safetensors" ]]; then
    echo "ERROR: CONTINUE_WEIGHTS=1 but no finetuned checkpoint found under $SOURCE_ROOT/checkpoints/"
    exit 1
  fi
  PRETRAINED="$LATEST_PRETRAINED"
  OUTPUT_DIR="${PI05_FINETUNE_CONTINUE_ROOT:-${SOURCE_ROOT}_continue}"
  if [[ -d "$OUTPUT_DIR" ]] && ! compgen -G "$OUTPUT_DIR/checkpoints/[0-9]*" > /dev/null 2>&1; then
    rm -rf "$OUTPUT_DIR"
  fi
  SOURCE_CFG="$LATEST_PRETRAINED/train_config.json"
  if [[ "$WANDB_ENABLE" == "1" && -f "$SOURCE_CFG" ]]; then
    WANDB_RUN_ID_OVERRIDE=$("$PYTHON" - <<PY
import json
print(json.load(open("$SOURCE_CFG")).get("wandb", {}).get("run_id") or "")
PY
    )
  fi
fi

echo "===== Step 3: finetune pi05_libero ====="
echo "  pretrained: $PRETRAINED"
echo "  dataset:    $PI05_LEROBOT_DATASET"
echo "  output:     $OUTPUT_DIR"
echo "  steps:      $STEPS  batch: $BATCH_SIZE  save_freq: $SAVE_FREQ  gpu: $CUDA_VISIBLE_DEVICES  resume: $RESUME"
if [[ -n "$TARGET_LOSS" ]]; then
  echo "  target_loss: $TARGET_LOSS (stop when reached)"
fi
if [[ "$CONTINUE_WEIGHTS" == "1" ]]; then
  echo "  continue from weights: $PRETRAINED (optimizer state not restored)"
  echo "  continue output dir:   $OUTPUT_DIR"
fi

RESUME_ARG=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARG=(--resume true)
  CONFIG_PATH=$(ls "$OUTPUT_DIR"/checkpoints/[0-9]*/pretrained_model/train_config.json 2>/dev/null | sort -V | tail -1 || true)
  if [[ -z "$CONFIG_PATH" ]]; then
    echo "ERROR: RESUME=1 but no train_config.json found under $OUTPUT_DIR/checkpoints/"
    exit 1
  fi
  RESUME_ARG+=(--config_path="$CONFIG_PATH")
  if [[ "$CKPT_PRUNE" == "1" && -d "$OUTPUT_DIR/checkpoints" ]]; then
    mapfile -t ALL_CKPTS < <(ls -d "$OUTPUT_DIR"/checkpoints/[0-9]*/ 2>/dev/null | sort -V)
    N=${#ALL_CKPTS[@]}
    if (( N > CKPT_KEEP )); then
      for (( i=0; i < N - CKPT_KEEP; i++ )); do
        echo "  prune old checkpoint: ${ALL_CKPTS[$i]}"
        rm -rf "${ALL_CKPTS[$i]}"
      done
    fi
  fi
elif [[ -d "$OUTPUT_DIR" ]]; then
  if compgen -G "$OUTPUT_DIR/checkpoints/*" > /dev/null 2>&1; then
    echo "Found existing checkpoints under $OUTPUT_DIR; set RESUME=1 to continue."
    exit 1
  fi
  # Empty or analysis-only leftovers: safe to remove for cold start.
  rm -rf "$OUTPUT_DIR"
  mkdir -p "${PERIODIC_ANALYSIS_DIR}"
  rm -f "${PERIODIC_ANALYSIS_DIR}/.processed_checkpoints.json"
  if [[ -f "${PERIODIC_ANALYSIS_DIR}/similarity_timeline.json" ]]; then
    mv "${PERIODIC_ANALYSIS_DIR}/similarity_timeline.json" \
      "${PERIODIC_ANALYSIS_DIR}/similarity_timeline.json.bak.$(date +%s)"
  fi
fi

WANDB_ARGS=(--wandb.enable=false)
if [[ "$WANDB_ENABLE" == "1" ]]; then
  WANDB_ARGS=(--wandb.enable=true --wandb.project="$WANDB_PROJECT" --wandb.disable_artifact=true)
  if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb.entity="$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_RUN_ID_OVERRIDE" ]]; then
    WANDB_ARGS+=(--wandb.run_id="$WANDB_RUN_ID_OVERRIDE")
  fi
fi

LEROBOT_TRAIN=$("$PYTHON" -c "import shutil; print(shutil.which('lerobot-train') or '')" 2>/dev/null || true)
if [[ -z "$LEROBOT_TRAIN" ]]; then
  TRAIN_CMD=("$PYTHON" -m lerobot.scripts.lerobot_train)
else
  TRAIN_CMD=("$LEROBOT_TRAIN")
fi

WATCH_PID=""
cleanup_watch() {
  if [[ -n "$WATCH_PID" ]] && kill -0 "$WATCH_PID" 2>/dev/null; then
    kill "$WATCH_PID" 2>/dev/null || true
    wait "$WATCH_PID" 2>/dev/null || true
  fi
}
trap cleanup_watch EXIT

if [[ "$SIM_WATCH" == "1" ]]; then
  mkdir -p "${PERIODIC_ANALYSIS_DIR}"
  echo "===== Step 3a: start periodic vision-similarity watcher (GPU ${SIM_CUDA_VISIBLE_DEVICES}) ====="
  WATCH_WANDB_ARGS=()
  if [[ "$WANDB_ENABLE" == "1" ]]; then
    WATCH_WANDB_ARGS=(--wandb_enable --wandb_project "$WANDB_PROJECT" --wandb_job_name pi05_study_scene4)
    if [[ -n "$WANDB_ENTITY" ]]; then
      WATCH_WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
    fi
  fi
  CUDA_VISIBLE_DEVICES="${SIM_CUDA_VISIBLE_DEVICES}" "$PYTHON" "${SCRIPT_DIR}/pi05_periodic_similarity_watch.py" \
    --train_output_dir "$OUTPUT_DIR" \
    --analysis_dir "$PERIODIC_ANALYSIS_DIR" \
    --lerobot_home "$HF_LEROBOT_HOME" \
    --repo_id "$LEROBOT_REPO" \
    --poll_sec "$SIM_POLL_SEC" \
    --device cuda:0 \
    --feature_types "vision_tower,vlm_prefix_l18" \
    --prune_checkpoints \
    --keep_checkpoints "$CKPT_KEEP" \
    ${WATCH_WANDB_ARGS[@]+"${WATCH_WANDB_ARGS[@]}"} \
    > "${PERIODIC_ANALYSIS_DIR}/watch.log" 2>&1 &
  WATCH_PID=$!
  echo "  watcher PID=$WATCH_PID log=$PERIODIC_ANALYSIS_DIR/watch.log wandb=${WANDB_ENABLE}"
fi

TRAIN_LOG="${PERIODIC_ANALYSIS_DIR}/train.log"
mkdir -p "${PERIODIC_ANALYSIS_DIR}"

if [[ -n "$TARGET_LOSS" ]]; then
  echo "===== Training (background) -> $TRAIN_LOG ====="
  "${TRAIN_CMD[@]}" \
    --dataset.repo_id="$LEROBOT_REPO" \
    --dataset.root="$PI05_LEROBOT_DATASET" \
    --policy.type=pi05 \
    --policy.pretrained_path="$PRETRAINED" \
    --policy.repo_id=local/pi05_study_scene4_finetuned \
    --policy.push_to_hub=false \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --output_dir="$OUTPUT_DIR" \
    --job_name=pi05_study_scene4 \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --eval_freq=0 \
    --save_freq="$SAVE_FREQ" \
    --log_freq=50 \
    "${WANDB_ARGS[@]}" \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    "${RESUME_ARG[@]}" >> "$TRAIN_LOG" 2>&1 &
  TRAIN_PID=$!
  echo "===== Loss monitor: stop when loss <= $TARGET_LOSS (train PID=$TRAIN_PID) ====="
  if "$PYTHON" "${SCRIPT_DIR}/pi05_loss_monitor.py" \
    --log "$TRAIN_LOG" \
    --target_loss "$TARGET_LOSS" \
    --train_pid "$TRAIN_PID" \
    --poll_sec 60; then
    echo "Target loss reached; stopping training PID=$TRAIN_PID"
    kill -INT "$TRAIN_PID" 2>/dev/null || kill "$TRAIN_PID" 2>/dev/null || true
  fi
  wait "$TRAIN_PID" 2>/dev/null || true
else
  "${TRAIN_CMD[@]}" \
    --dataset.repo_id="$LEROBOT_REPO" \
    --dataset.root="$PI05_LEROBOT_DATASET" \
    --policy.type=pi05 \
    --policy.pretrained_path="$PRETRAINED" \
    --policy.repo_id=local/pi05_study_scene4_finetuned \
    --policy.push_to_hub=false \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --output_dir="$OUTPUT_DIR" \
    --job_name=pi05_study_scene4 \
    --steps="$STEPS" \
    --batch_size="$BATCH_SIZE" \
    --eval_freq=0 \
    --save_freq="$SAVE_FREQ" \
    --log_freq=50 \
    "${WANDB_ARGS[@]}" \
    --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}' \
    "${RESUME_ARG[@]}"
fi

CKPT=$(ls -d "$OUTPUT_DIR"/checkpoints/*/ 2>/dev/null | sort -V | tail -1)
if [[ -z "$CKPT" ]]; then
  CKPT="$OUTPUT_DIR"
fi
CKPT_PRETRAINED="${CKPT}"
if [[ -f "${CKPT}/pretrained_model/model.safetensors" ]]; then
  CKPT_PRETRAINED="${CKPT}/pretrained_model"
fi
echo "$CKPT" > "$OUTPUT_DIR/latest_checkpoint.txt"
echo "===== Finetune done. Checkpoint: $CKPT_PRETRAINED ====="

if [[ "$SKIP_FINAL_ANALYSIS" == "1" ]]; then
  echo "SKIP_FINAL_ANALYSIS=1; skipping final feature extraction."
  echo "All done."
  echo "  finetune ckpt: $CKPT_PRETRAINED"
  exit 0
fi

echo "===== Step 4: extract features + compare similarity (finetuned ckpt) ====="
for FT in vision_tower vlm_prefix_l18; do
  "$PYTHON" "${SCRIPT_DIR}/extract_pi05_vision_features.py" \
    --checkpoint "$CKPT_PRETRAINED" \
    --lerobot_home "$HF_LEROBOT_HOME" \
    --repo_id "$LEROBOT_REPO" \
    --output_dir "$ANALYSIS_DIR/vision_features_finetuned/${FT}" \
    --device "cuda:0" \
    --feature_type "$FT"
  "$PYTHON" "${SCRIPT_DIR}/compare_pi05_vision_features.py" \
    --feature_dir "$ANALYSIS_DIR/vision_features_finetuned/${FT}"
done

echo "All done."
echo "  finetune ckpt: $CKPT_PRETRAINED"
echo "  vision feats:  $ANALYSIS_DIR/vision_features_finetuned/{vision_tower,vlm_prefix_l18}/"
if [[ "$SIM_WATCH" == "1" ]]; then
  echo "  periodic sim:  $PERIODIC_ANALYSIS_DIR/similarity_timeline.json"
fi
