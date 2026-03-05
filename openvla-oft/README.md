CUDA_VISIBLE_DEVICES=1 \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "/home/hongyi/16831pro_fine_tune/openvla-oft/openvla-7b-oft-finetuned-libero-goal" \
  --data_root_dir "/home/hongyi/16831pro_fine_tune/openvla-oft/datasets/modified_libero_rlds" \
  --dataset_name "libero_goal_no_noops" \
  --run_root_dir "/home/hongyi/runs/openvla" \
  --use_lora True \
  --lora_rank 32 \
  --batch_size 4 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project "openvla" \
  --wandb_entity "maggiesh" \
  --save_freq 50
  --resume True





python ../tools/mask_processor.py


deattatch: Ctrl + B 然后 D
tmux attach -t vla-oft

阅读：Ctrl + B 然后 [



   CUDA_VISIBLE_DEVICES=1 \
    torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "/home/hongyi/runs/openvla/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--400_chkpt" \
    --data_root_dir "/home/hongyi/16831pro_fine_tune/openvla-oft/datasets/modified_libero_rlds" \
    --dataset_name "libero_goal_no_noops" \
    --run_root_dir "/home/hongyi/runs/openvla" \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 4 \
    --grad_accumulation_steps 1 \
    --learning_rate 5e-4 \
    --image_aug True \
    --wandb_project "openvla" \
    --wandb_entity "maggiesh" \
    --save_freq 50 \
    --resume True \
    --resume_step 400


chmod +x auto_run.sh
./auto_run.sh


CUDA_VISIBLE_DEVICES=1 \

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "/home/hongyi/runs/openvla/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--450_chkpt" \
  --task_suite_name libero_goal


PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH \
python experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "/home/ubuntu/runs/openvla/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--1700_chkpt" \
  --task_suite_name libero_goal

git pull

git add .
git commit -m "add_rlds_processor"
git push


python tools/rlds_mask.py --data_mix libero_goal_no_noops --debug_dir rlds_mask_debug --debug_every 200 --no_mask_wrist 

./auto_run.sh --fast_model



  export ROBOFLOW_API_KEY="

CKPT_DIR="/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt"

PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH \
python experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --base_vla_path "/home/ubuntu/runs/openvla/openvla-7b" \
  --use_proprio True \
  --task_suite_name libero_goal



PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --base_vla_path "/home/ubuntu/runs/openvla/openvla-7b" \
  --use_proprio True \
  --task_suite_name libero_goal \
  --load_in_8bit True


./auto_run.sh --fast_model


cd /home/ubuntu/16831pro_fine_tune/openvla-oft
PYTHONPATH=. /home/ubuntu/miniconda3/envs/vla-preprocess/bin/python test_roboflow_gripper.py



PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_eval_mask.py \
  --pretrained_checkpoint "/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt" \
  --base_vla_path /home/ubuntu/16831pro_fine_tune/openvla-oft/checkpoints/openvla-7b \
  --use_proprio True \
  --task_suite_name libero_goal \
  --perturb_colors False \
  --load_in_8bit True \
  --perturb_bowl True
  



cd /home/ubuntu/16831pro_fine_tune/openvla-oft
PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_generalization_eval.py \
  --pretrained_checkpoint "/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt" \
  --base_vla_path "/home/ubuntu/16831pro_fine_tune/openvla-oft/checkpoints/openvla-7b" \
  --load_in_8bit True



cd /home/ubuntu/16831pro_fine_tune/openvla-oft
PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_background_perturb_eval.py \
  --pretrained_checkpoint "/home/ubuntu/runs/openvla_adapters/openvla-7b+libero_goal_no_noops+b8+lr-0.0001+lora-r8+dropout-0.0+lora-attn-only--13500_chkpt" \
  --base_vla_path "/home/ubuntu/16831pro_fine_tune/openvla-oft/checkpoints/openvla-7b" \
  --load_in_8bit True






cd /home/ubuntu/16831pro_fine_tune/openvla-oft
conda activate simplevla
PYTHONPATH=. python scripts/extract_hidden_states.py \
  --image_path /home/ubuntu/16831pro_fine_tune/zt/test_book.png \
  --output_dir /home/ubuntu/16831pro_fine_tune/zz/output_hidden_states












export ROBOFLOW_API_KEY="


  cd 16831pro_fine_tune/openvla-oft/mask_processor.py

   python experiments/robot/libero/run_single_chunk_inference.py \
   --proprio "0.671691520561037,0.016602214,0.048589394,0.910049785,0.538836866,0.277672637,0.954056705,-0.96734364
" \
   --load_in_8bit True


./experiments/robot/libero/run_all_bg_perturb.sh





./experiments/robot/libero/run_all_color_perturb.sh



PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "moojink/openvla-7b-oft-finetuned-libero-goal" \
  --task_suite_name libero_goal \
  --num_images_in_input 2 \
  --num_trials_per_task 10 \
  --task_ids "5,8" \
  --load_in_8bit True














cd /home/ubuntu/16831pro_fine_tune/SimpleVLA-RL
sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
export RAY_memory_monitor_refresh_ms=0
bash examples/run_openvla_oft_rl_libero_push_perturb.sh



/ubuntu/16831pro_fine_tune/openvla-oft
PYTHONPATH=. python scripts/extract_hidden_states.py \
  --image_path /home/ubuntu/16831pro_fine_tune/zt/test_book.png \
  --load_in_4bit



# Put bowl on plate 评测（无移位、无 plate 环境，mask 目标区为白色）
# 不需要移位时直接跑下面即可（默认 use_no_plate_env=True, dest_mask_white=True）
PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_put_bowl_shifted_mask_eval.py \
  --pretrained_checkpoint "$CURRENT_CHECKPOINT" \
  --base_vla_path "${OPENVLA_OFT_ROOT}/checkpoints/openvla-7b" \
  --model_label "current" \
  --use_mask_for_policy True \
  --num_trials_per_task 3
# 可选：有 plate+移位 时加 --use_no_plate_env False --shift_plate_pixels 80


  PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH python experiments/robot/libero/run_libero_push_stove_mask_eval.py \
  --use_mask_for_policy True --num_trials_per_task 3






  cd /home/ubuntu/16831pro_fine_tune/openvla-oft
export PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH
python experiments/robot/libero/run_libero_put_bowl_shifted_mask_eval.py \
  --pretrained_checkpoint "$CURRENT_CHECKPOINT" \
  --base_vla_path "${OPENVLA_OFT_ROOT}/checkpoints/openvla-7b" \
  --model_label "current" \
  --use_mask_for_policy True \
  --num_trials_per_task 3




  cd /home/ubuntu/16831pro_fine_tune/openvla-oft
export PYTHONPATH=/home/ubuntu/16831pro_fine_tune/LIBERO:$PYTHONPATH
python experiments/robot/libero/run_libero_put_bowl_shifted_mask_eval.py \
  --pretrained_checkpoint "$CURRENT_CHECKPOINT" \
  --base_vla_path "${OPENVLA_OFT_ROOT}/checkpoints/openvla-7b" \
  --model_label "current" \
  --use_mask_for_policy True \
  --num_trials_per_task 3