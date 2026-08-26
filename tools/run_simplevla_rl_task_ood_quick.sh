#!/usr/bin/env bash
# Quick SimpleVLA-RL on one LIBERO goal task + color OOD (plate|bowl).
# Saves LoRA adapter only (no merge). Env knobs: TASK_ID, PERTURB_MODE, NOTE, GPU, VAL_ONLY.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SVLA="${ROOT}/SimpleVLA-RL"
STORAGE="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE}/conda_envs/simplevla/bin/python}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-${STORAGE}/ckpts/Openvla-oft-SFT-libero-goal-traj1}"
CKPT_PATH="${CKPT_PATH:-${STORAGE}/runs/simplevla_rl_ood}"
NOTE="${NOTE:-$(date +%Y%m%d_%H%M%S)}"
TASK_ID="${TASK_ID:-5}"
PERTURB_MODE="${PERTURB_MODE:-plate}"  # plate | bowl | none
DATASET_NAME="${DATASET_NAME:-libero_goal}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU="${GPU:-0,1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
VAL_ONLY="${VAL_ONLY:-False}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-t${TASK_ID}_${PERTURB_MODE}_${NOTE}}"
# Per-run align.json so parallel jobs do not overwrite CUDA_VISIBLE_DEVICES
ALIGN_PATH="${CKPT_PATH}/align_${EXPERIMENT_NAME}.json"

mkdir -p "${CKPT_PATH}" "${ROOT}/openvla-oft/logs"
LOG="${ROOT}/openvla-oft/logs/simplevla_rl_${EXPERIMENT_NAME}.log"
cp -f "${SVLA}/align.json" "${ALIGN_PATH}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-local}"
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export RAY_memory_usage_threshold=0.99
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES="${GPU}"
export ROBOT_PLATFORM=LIBERO
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export SIMPLEVAL_RL_SAVE_MERGED=0
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${SVLA}:${ROOT}/LIBERO:${ROOT}/openvla-oft:${PYTHONPATH:-}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${EXPERIMENT_NAME}}"
mkdir -p "${RAY_TMPDIR}"

python3 - <<PY
import json
p="${ALIGN_PATH}"
d=json.load(open(p))
d.setdefault("env_vars",{})
d["env_vars"]["WANDB_API_KEY"]=d["env_vars"].get("WANDB_API_KEY") or "local"
d["env_vars"]["WANDB_MODE"]="offline"
d["env_vars"]["TOKENIZERS_PARALLELISM"]="true"
d["env_vars"]["ROBOT_PLATFORM"]="LIBERO"
d["env_vars"]["MUJOCO_GL"]="${MUJOCO_GL:-egl}"
d["env_vars"]["PYOPENGL_PLATFORM"]="${PYOPENGL_PLATFORM:-egl}"
d["env_vars"]["CUDA_VISIBLE_DEVICES"]="${GPU}"
d["env_vars"]["PYTHONPATH"]="${SVLA}:${ROOT}/LIBERO:${ROOT}/openvla-oft"
d["env_vars"]["TF_CPP_MIN_LOG_LEVEL"]="2"
json.dump(d, open(p,"w"), indent=2)
print("updated", p)
PY

bash "${SVLA}/examples/overwrite_vla_ckpt_utils.sh" "${SFT_MODEL_PATH}"

echo "===== $(date -Iseconds) SimpleVLA-RL OOD start =====" | tee "${LOG}"
echo "SFT=${SFT_MODEL_PATH}" | tee -a "${LOG}"
echo "OUT=${CKPT_PATH}/RL/${EXPERIMENT_NAME}" | tee -a "${LOG}"
echo "GPU=${GPU} task_id=${TASK_ID} perturb=${PERTURB_MODE} val_only=${VAL_ONLY} epochs=${TOTAL_EPOCHS}" | tee -a "${LOG}"

cd "${SVLA}"
HYDRA_FULL_ERROR=1 "${PYTHON}" -u -m verl.trainer.main_ppo \
  data.task_suite_name=${DATASET_NAME} \
  data.num_trials_per_task=4 \
  data.libero_single_task_id=${TASK_ID} \
  data.n_samples=4 \
  data.filter_accuracy=False \
  data.oversample_factor=1 \
  data.train_batch_size=2 \
  data.val_batch_size=4 \
  data.max_prompt_length=256 \
  data.max_response_length=128 \
  actor_rollout_ref.model.path=${SFT_MODEL_PATH} \
  actor_rollout_ref.model.vla=openvla-oft \
  actor_rollout_ref.model.action_token_len=7 \
  actor_rollout_ref.model.action_chunks_len=8 \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj] \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.optim.warmup_style=constant \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size=${NUM_GPUS} \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.grad_clip=1 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.num_images_in_input=1 \
  actor_rollout_ref.actor.traj_mini_batch_size=2 \
  actor_rollout_ref.actor.entropy_coeff=0. \
  actor_rollout_ref.rollout.num_images_in_input=1 \
  actor_rollout_ref.rollout.use_proprio=False \
  actor_rollout_ref.rollout.val_micro_batch_size=2 \
  actor_rollout_ref.rollout.temperature=0.5 \
  actor_rollout_ref.rollout.experiment_name=${EXPERIMENT_NAME} \
  actor_rollout_ref.rollout.micro_batch_size=1 \
  actor_rollout_ref.rollout.unnorm_key=libero_goal_no_noops \
  actor_rollout_ref.rollout.model_family=openvla \
  actor_rollout_ref.rollout.task_suite_name=${DATASET_NAME} \
  actor_rollout_ref.rollout.num_steps_wait=10 \
  actor_rollout_ref.rollout.pretrained_checkpoint=${SFT_MODEL_PATH} \
  actor_rollout_ref.rollout.center_crop=True \
  actor_rollout_ref.rollout.perturb_colors=True \
  actor_rollout_ref.rollout.perturb_mode=${PERTURB_MODE} \
  actor_rollout_ref.rollout.max_prompt_length=512 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=hf \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=grpo \
  algorithm.adv_params.verifier_gamma=1.0 \
  algorithm.adv_params.reward_model_gamma=1.0 \
  trainer.logger=['console'] \
  trainer.project_name=SimpleVLA-RL \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.default_local_dir=${CKPT_PATH}/RL/${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=${NUM_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.val_only=${VAL_ONLY} \
  trainer.runtime_env=${ALIGN_PATH} \
  trainer.wandb_mode=offline \
  trainer.val_before_train=True \
  2>&1 | tee -a "${LOG}"

echo "===== $(date -Iseconds) DONE rc=${PIPESTATUS[0]} =====" | tee -a "${LOG}"
echo "LoRA dirs:" | tee -a "${LOG}"
find "${CKPT_PATH}/RL/${EXPERIMENT_NAME}" -type d -name lora_adapter 2>/dev/null | tee -a "${LOG}"
