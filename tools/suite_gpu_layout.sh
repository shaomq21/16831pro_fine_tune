#!/usr/bin/env bash
# Default multi-GPU layout per masked LIBERO suite (override via env before launch).
# Each suite uses torchrun --nproc-per-node NUM_GPUS with CUDA_VISIBLE_DEVICES.

case "${SUITE:?set SUITE}" in
  spatial)
    export CUDA_VISIBLE_DEVICES="0,2"
    export NUM_GPUS="2"
    ;;
  goal)
    export CUDA_VISIBLE_DEVICES="4,5"
    export NUM_GPUS="2"
    ;;
  object)
    export CUDA_VISIBLE_DEVICES="6,7"
    export NUM_GPUS="2"
    ;;
  study_scene4)
    # Default: GPU 3 solo (1/3 often occupied by other jobs). Override via finetune_suite_study_scene4_gpu.env
    export CUDA_VISIBLE_DEVICES="3"
    export NUM_GPUS="1"
    ;;
  *)
    echo "Unknown SUITE=${SUITE}" >&2
    exit 1
    ;;
esac
