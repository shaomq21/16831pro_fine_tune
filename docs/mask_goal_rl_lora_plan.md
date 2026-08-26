# Mask-model RL LoRA plan (after goal ckpt arrives)

## Done now
- Stopped all SimpleVLA-RL / bowl traj1 runs
- Deleted `ckpts/Openvla-oft-SFT-libero-goal-traj1` (~15G freed; disk ~85G avail)

## Important compatibility
Mask goal policy is **continuous OFT** (`action_head` + LoRA on `openvla-7b`), **not** SimpleVLA discrete-token SFT.
→ Official SimpleVLA-RL `main_ppo` path cannot load it as-is.
→ RL will use the continuous LoRA recipe (`tools/run_rl_lora_color_quick.sh` / GRPO-style), referencing SimpleVLA-RL’s short-run / unmerged-LoRA setup.

## Existing baselines (skip re-eval) — `ours_masked` goal

| Task | Condition | SR | Source |
|------|-----------|-----|--------|
| #8 put bowl on plate | color (variants 0/1, 5 trials each) | **40%** | `goal_perturb_matrix/.../ours_color_matrix_20260721` |
| #5 push plate | (prior continuous RL run) | ~25–40% color | earlier `goal_grpo` / matrix |

**Primary pick:** task **8** color (40% mid, non-zero) → train RL LoRA unmerged → target higher SR.  
**Secondary:** task **5** plate color if time.

## Ckpt expected path (upload in progress)

```
.../runs/openvla_adapters/openvla-7b+dual_masked_goal+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_goal_oft_lr/
```

Currently has `lora_adapter/`, `action_head--latest_checkpoint.pt`, and `model-00001-of-00004.safetensors.filepart` (upload).

## After ckpt complete
1. `SUITE=goal TASK_ID=8 NOTE=mask_bowl8 ... bash tools/run_rl_lora_color_quick.sh` (LoRA only, no merge)
2. Optionally parallel task 5 on another GPU
3. Append results to `docs/simplevla_rl_ood_lora_results.md` (mask section)
4. `val_before_train` / baseline eval **skipped** (use table above)

## Stability knobs (avoid prior 40%→0% collapse)
- Smaller lr, freeze or low lr on vision LoRA if needed
- Only update groups with ≥1 success
- Short iters (≤8–10), save unmerged RL LoRA each step
