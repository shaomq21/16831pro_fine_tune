# SimpleVLA-RL OOD LoRA Results (unmerged)

Framework: [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)  
SFT backbone: `Haozhan72/Openvla-oft-SFT-libero-goal-traj1`  
Policy save: **LoRA adapter only** (`SIMPLEVAL_RL_SAVE_MERGED=0`, no full merge)

Recipe (shared): 2×GPU FSDP, LoRA r=16 on `q/k/v/o_proj`, GRPO `n_samples=4`, `temperature=0.5`, `val_before_train=True`, short epochs.

Launcher: `tools/run_simplevla_rl_task_ood_quick.sh`  
(`TASK_ID`, `PERTURB_MODE=plate|bowl`, `NOTE`, `GPU`, `VAL_ONLY`)

---

## Summary table

| Task | OOD | Baseline val | Best RL val | Best LoRA | Status |
|------|-----|--------------|-------------|-----------|--------|
| goal#5 *push the plate to the front of the stove* | plate color (tint 0.35) | **25%** | **75%** | `.../push_color_lora_v15/.../global_step_10/lora_adapter` | **done** |
| goal#8 *put the bowl on the plate* | bowl color (tint 0.60) | **50%** | **75%** | `.../t8_bowl_rl3_bowl8/.../global_step_3/lora_adapter` | **done** (run ended early after step4; best already logged) |
| goal#6 *put the cream cheese in the bowl* | bowl color (tint 0.60) | **0%** | **25%** (step5) | `.../t6_bowl_rl3_bowl6/.../global_step_5/lora_adapter` | exploratory (baseline was 0; mid-run lift only) |

---

## Experiment A — Push plate + plate color OOD

| Field | Value |
|-------|--------|
| Suite / task | `libero_goal` task **5** |
| OOD | Plate recolor via red-ring ROI → yellow/cyan (`tint_a=0.35`) |
| Run | `push_color_lora_v15` |

### Success rate (val, 4 episodes; greedy)

| Checkpoint | Val SR |
|------------|--------|
| Baseline (`val_before_train`) | **25%** |
| Best (`global_step_10`) | **75%** |
| Also ≥50% | steps 1, 2, 4, 5, 9 |

```
/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/simplevla_rl_push_color/RL/push_color_lora_v15/actor/global_step_10/lora_adapter/
```

---

## Experiment B — Put bowl on plate + bowl color OOD

| Field | Value |
|-------|--------|
| Suite / task | `libero_goal` task **8** |
| OOD | Dark-grey bowl → red/blue soft tint (`tint_a=0.60`) |
| Run | `t8_bowl_rl3_bowl8` |

### Success rate (val, 4 episodes; greedy)

| Checkpoint | Val SR |
|------------|--------|
| Baseline | **50%** |
| step 1–2 | 50% / 25% |
| Best (`global_step_3`, `4`) | **75%** |

```
/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/simplevla_rl_ood/RL/t8_bowl_rl3_bowl8/actor/global_step_3/lora_adapter/
```

Note: training aborted after step 4 (`DONE rc=1`); LoRA adapters through step 4 remain on disk. Best val already **75%**.

---

## Experiment C — Cream cheese in bowl (exploratory)

Baseline under the same bowl tint was **0%** (too hard / task sensitive). Val briefly reached **25%** at step 5, then returned to 0% by step 6. Not used as a primary positive case because the selection criterion was **non-zero but not-high** baseline SR.

---

## Probe notes (task selection)

Soft bowl tint `0.50`: task1≈100%, task4≈75%, task8≈75% (too easy for clear RL headroom).  
Hard tint `0.85`: several tasks → 0%.  
Tint `0.60` on task8 → **50%** baseline (kept).

---

## How to reproduce

```bash
# Task 5 plate color
NOTE=v15 GPU=0,1 NUM_GPUS=2 TASK_ID=5 PERTURB_MODE=plate \
  bash tools/run_simplevla_rl_task_ood_quick.sh

# Task 8 bowl color
NOTE=rl3_bowl8 GPU=0,1 NUM_GPUS=2 TASK_ID=8 PERTURB_MODE=bowl TOTAL_EPOCHS=6 \
  bash tools/run_simplevla_rl_task_ood_quick.sh
```

Logs: `openvla-oft/logs/simplevla_rl_*.log`

---

## Mask-model continuous RL (dual_masked_goal)

### First attempt — failed (collapsed)
| Run | Task | Matrix baseline | Final |
|-----|------|-----------------|-------|
| `mask_bowl8_color` / `mask_plate5_color` | #8 / #5 | 40% / mid | **0%** after first update |

### Stable recipe — multi-task (mask goal)

Recipe: only groups with ≥1 success; BC success trajs only; freeze vision LoRA (action_head only); smaller lr; rollback on collapse; skip baseline eval.

| Run | Task | Known baseline (matrix) | Final greedy | Verdict |
|-----|------|-------------------------|--------------|---------|
| `mask_bowl8_stable` | #8 put bowl on plate + bowl | **40%** | **87.5%** | ✅ clear lift |
| `stable_t4` | #4 bowl on cabinet + bowl | mid (plate-color was high) | **87.5%** | ✅ strong |
| `mask_plate5_stable` | #5 push plate + plate color | **~33%** | **37.5%** | ~ mild |
| `stable_t6` | #6 cream cheese + color | **~17%** | **25%** | ~ mild |
| `stable_t1` | #1 bowl on stove + bowl | ~40% (other evals) | **12.5%** | ❌ weak / mismatch |
| `stable_t2` | #2 wine on cabinet + color | ~83% | **0%** | ❌ eval collapse |
| `stable_t3` | #3 drawer+bowl + bowl | color **0%** | — | killed (no successes) |

```
.../runs/rl_lora_color_quick/libero_goal_task8_mask_bowl8_stable/
.../runs/rl_lora_color_quick/libero_goal_task4_stable_t4/
.../runs/rl_lora_color_quick/libero_goal_task5_mask_plate5_stable/
```

### Key engineering fixes

1. Env workers: `multiprocessing` **spawn**  
2. TF image ops on **CPU**  
3. GRPO: correct group `uid` + `std`  
4. LoRA-only save; avoid shadowing `torch` via local `import torch.distributed`  
5. Plate ROI by red ring; bowl via dark-grey mask + soft tint  
6. Per-run `align_*.json` + `RAY_TMPDIR` for parallel jobs  
