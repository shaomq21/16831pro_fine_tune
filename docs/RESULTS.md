> GitHub copy of `openvla-oft/runs/all_suites_perturb_matrix/summary/RESULTS.md`.
> Rollout matrix videos: [`scenario_grids/`](./scenario_grids/).

# Dual-masked OpenVLA eval results (consolidated)

- Generated: 2026-07-27T15:15:42.310937+00:00
- Model: ours dual-masked LoRA **after rescue finetune** (LoRA auto-merged at eval)
- Scope: rescued / previously-low tasks only (not full suite matrix)
- Run: `rescue_reeval_20260726_032817` + color rerun `color_rerun_20260727_013826`
- Videos: `openvla-oft/rollouts/2026_07_26/` and `2026_07_27/`

## 0. Post-rescue re-eval (origin / lang / bg / color)

**Lang method (预定):** masked instruction 不变；每 task **1× held-out baseline** + **4× 从剩余 trial 随机抽** → mean-of-5；l1/l2 不同 seed。详见 `langshift_mean5_post_rescue.json`。

**`bg-pooled`：** origin + bg-0 + bg-1 + bg-2 的成功数合计（**不含** color-0/1）。

| Suite | origin | lang-l1 | lang-l2 | bg-0 | bg-1 | bg-2 | color-0 | color-1 | bg-pooled |
|-------|--------|---------|---------|------|------|------|---------|---------|-----------|
| goal | 16/24 (67%) | 68% | 65% | 16/24 (67%) | 16/24 (67%) | 17/24 (71%) | 16/24 (67%) | 15/24 (62%) | 65/96 (68%) |
| object | 9/18 (50%) | 53% | 63% | 8/18 (44%) | 9/18 (50%) | 8/18 (44%) | 12/18 (67%) | 11/18 (61%) | 34/72 (47%) |
| spatial | 18/27 (67%) | 64% | 53% | 15/27 (56%) | 14/27 (52%) | 15/27 (56%) | 14/27 (52%) | 13/27 (48%) | 62/108 (57%) |
| study_scene4 | 7/12 (58%) | 70% | 60% | 8/12 (67%) | 8/12 (67%) | 8/12 (67%) | 8/12 (67%) | 9/12 (75%) | 31/48 (65%) |

### Language-shift detail (post-rescue mean-of-5)

| Suite | Task | new | lang-l1 | lang-l2 |
|-------|------|-----|---------|---------|
| goal | open the middle drawer of the cabinet | 1 | 80% | 100% |
| goal | open the top drawer and put the bowl inside | 0 | 40% | 20% |
| goal | push the plate to the front of the stove | 0 | 40% | 40% |
| goal | put the bowl on top of the cabinet | 0 | 60% | 80% |
| goal | put the cream cheese in the bowl | 0 | 20% | 0% |
| goal | put the wine bottle on the rack | 1 | 100% | 100% |
| goal | put the wine bottle on top of the cabinet | 1 | 100% | 80% |
| goal | turn on the stove | 1 | 100% | 100% |
| object | pick up the alphabet soup and place it in the basket | 0 | 20% | 60% |
| object | pick up the chocolate pudding and place it in the basket | 1 | 80% | 80% |
| object | pick up the cream cheese and place it in the basket | 1 | 40% | 40% |
| object | pick up the ketchup and place it in the basket | 1 | 60% | 80% |
| object | pick up the milk and place it in the basket | 1 | 20% | 40% |
| object | pick up the tomato sauce and place it in the basket | 1 | 100% | 80% |
| spatial | pick up the black bowl from table center and place it on ... | 1 | 100% | 60% |
| spatial | pick up the black bowl in the top drawer of the wooden ca... | 1 | 60% | 20% |
| spatial | pick up the black bowl next to the cookie box and place i... | 1 | 40% | 80% |
| spatial | pick up the black bowl next to the plate and place it on ... | 1 | 80% | 60% |
| spatial | pick up the black bowl next to the ramekin and place it o... | 1 | 80% | 100% |
| spatial | pick up the black bowl on the cookie box and place it on ... | 0 | 0% | 0% |
| spatial | pick up the black bowl on the ramekin and place it on the... | 1 | 60% | 40% |
| spatial | pick up the black bowl on the stove and place it on the p... | 1 | 80% | 40% |
| spatial | pick up the black bowl on the wooden cabinet and place it... | 1 | 80% | 80% |
| study_scene4 | pick up the book in the middle and place it on the cabine... | 0 | 0% | 0% |
| study_scene4 | pick up the book on the left and place it on top of the s... | 1 | 80% | 80% |
| study_scene4 | pick up the book on the right and place it on the cabinet... | 1 | 100% | 100% |
| study_scene4 | pick up the book on the right and place it under the cabi... | 1 | 100% | 60% |

### Per-task visual detail

#### goal

| Task | origin | bg-0 | bg-1 | bg-2 | color-0 | color-1 |
|------|--------|------|------|------|---------|---------|
| open the middle drawer of the cabinet | 2/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |
| open the top drawer and put the bowl inside | 1/3 | 0/3 | 1/3 | 1/3 | 0/3 | 0/3 |
| push the plate to the front of the stove | 1/3 | 1/3 | 1/3 | 2/3 | 1/3 | 1/3 |
| put the bowl on top of the cabinet | 2/3 | 2/3 | 1/3 | 2/3 | 3/3 | 3/3 |
| put the cream cheese in the bowl | 1/3 | 1/3 | 1/3 | 0/3 | 1/3 | 0/3 |
| put the wine bottle on the rack | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| put the wine bottle on top of the cabinet | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 2/3 |
| turn on the stove | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

#### object

| Task | origin | bg-0 | bg-1 | bg-2 | color-0 | color-1 |
|------|--------|------|------|------|---------|---------|
| pick up the alphabet soup and place it in the basket | 1/3 | 1/3 | 1/3 | 1/3 | 2/3 | 0/3 |
| pick up the chocolate pudding and place it in the basket | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 |
| pick up the cream cheese and place it in the basket | 1/3 | 0/3 | 1/3 | 0/3 | 2/3 | 2/3 |
| pick up the ketchup and place it in the basket | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 | 2/3 |
| pick up the milk and place it in the basket | 1/3 | 0/3 | 0/3 | 0/3 | 2/3 | 1/3 |
| pick up the tomato sauce and place it in the basket | 2/3 | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 |

#### spatial

| Task | origin | bg-0 | bg-1 | bg-2 | color-0 | color-1 |
|------|--------|------|------|------|---------|---------|
| pick up the black bowl from table center and place it on the plate | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 1/3 |
| pick up the black bowl in the top drawer of the wooden cabinet and ... | 2/3 | 2/3 | 1/3 | 1/3 | 1/3 | 1/3 |
| pick up the black bowl next to the cookie box and place it on the p... | 2/3 | 2/3 | 1/3 | 1/3 | 3/3 | 2/3 |
| pick up the black bowl next to the plate and place it on the plate | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 1/3 |
| pick up the black bowl next to the ramekin and place it on the plate | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 |
| pick up the black bowl on the cookie box and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/3 |
| pick up the black bowl on the ramekin and place it on the plate | 2/3 | 2/3 | 1/3 | 1/3 | 0/3 | 0/3 |
| pick up the black bowl on the stove and place it on the plate | 3/3 | 1/3 | 1/3 | 3/3 | 2/3 | 2/3 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 2/3 | 1/3 | 3/3 | 2/3 | 2/3 | 1/3 |

#### study_scene4

| Task | origin | bg-0 | bg-1 | bg-2 | color-0 | color-1 |
|------|--------|------|------|------|---------|---------|
| pick up the book in the middle and place it on the cabinet shelf | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| pick up the book on the left and place it on top of the shelf | 2/3 | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 |
| pick up the book on the right and place it on the cabinet shelf | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| pick up the book on the right and place it under the cabinet shelf | 2/3 | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 |

---

## 0b. Generated-mask re-eval (match training non-simu backend)

- Model: same post-rescue dual-masked LoRA
- Mask source: **not** sim seg (`use_mask_from_env=False`); matches RLDS non-simu half
  - **goal** → Grounded-DINO + SAM1 (`sam_backend=sam1`)
  - **spatial** → SAM3 + temporal `EpisodeMaskTracker` (`sam_backend=sam3`)
  - object / study_scene4：训练无 simu mask → 未跑
- Schedule: origin + bg-0/1/2 + color-0/1，TRIALS=3；lang = mean-of-5（note=`sam_color_lang_20260728_031904`）
- Note: `sam_postrescue_20260727_151831`；spatial 中途 mask-worker timeout，缺 3 task 已 resume 补齐
- Logs: `openvla-oft/runs/sam_mask_reeval/logs/`
- Videos: `openvla-oft/rollouts/2026_07_27/` (`ours_masked_sam1` / `ours_masked_sam3`)

**Column note:** `bg-pooled` = **origin + bg-0 + bg-1 + bg-2** 成功数合计（不含 color），与 §0 主表同义。

| Suite | mask | origin | lang-l1 | lang-l2 | bg-0 | bg-1 | bg-2 | color-0 | color-1 | bg-pooled |
|-------|------|--------|---------|---------|------|------|------|---------|---------|-----------|
| goal | Grounded SAM (sam1) | 12/30 (40%) | 32% | 32% | 11/30 (37%) | 12/30 (40%) | 12/30 (40%) | 10/30 (33%) | 8/30 (27%) | 47/120 (39%) |
| spatial | SAM3 + temporal | 3/30 (10%) | 18% | 14% | 7/30 (23%) | 7/30 (23%) | 7/30 (23%) | 3/30 (10%) | 5/30 (17%) | 24/120 (20%) |



Compare (same schedule, simu mask on rescued-task subset in §0 vs full-suite generated mask here):

| Suite | simu-mask origin (§0, rescued tasks) | generated-mask origin (full suite) | generated bg-pooled |
|-------|--------------------------------------|------------------------------------|---------------------|
| goal | 16/24 (67%) | 12/30 (40%) | 47/120 (39%) |
| spatial | 18/27 (67%) | 3/30 (10%) | 24/120 (20%) |

### Per-task (generated mask)

#### goal — Grounded SAM

| Task | origin | bg-0 | bg-1 | bg-2 | pooled |
|------|--------|------|------|------|--------|
| open the middle drawer of the cabinet | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| open the top drawer and put the bowl inside | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| push the plate to the front of the stove | 2/3 | 1/3 | 1/3 | 2/3 | 6/12 |
| put the bowl on the plate | 3/3 | 3/3 | 3/3 | 3/3 | 12/12 |
| put the bowl on the stove | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| put the bowl on top of the cabinet | 2/3 | 2/3 | 3/3 | 2/3 | 9/12 |
| put the cream cheese in the bowl | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| put the wine bottle on the rack | 2/3 | 2/3 | 2/3 | 2/3 | 8/12 |
| put the wine bottle on top of the cabinet | 3/3 | 3/3 | 3/3 | 3/3 | 12/12 |
| turn on the stove | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |

#### spatial — SAM3 (main + resume merged)

| Task | origin | bg-0 | bg-1 | bg-2 | pooled |
|------|--------|------|------|------|--------|
| pick up the black bowl between the plate and the ramekin and place it on the plate | 0/3 | 1/3 | 2/3 | 2/3 | 5/12 |
| pick up the black bowl from table center and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| pick up the black bowl next to the cookie box and place it on the plate | 2/3 | 2/3 | 2/3 | 2/3 | 8/12 |
| pick up the black bowl next to the plate and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| pick up the black bowl next to the ramekin and place it on the plate | 0/3 | 1/3 | 2/3 | 0/3 | 3/12 |
| pick up the black bowl on the cookie box and place it on the plate | 0/3 | 1/3 | 1/3 | 1/3 | 3/12 |
| pick up the black bowl on the ramekin and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| pick up the black bowl on the stove and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/12 |
| pick up the black bowl on the wooden cabinet and place it on the plate | 1/3 | 2/3 | 0/3 | 2/3 | 5/12 |

---

## Archive: pre-rescue full-suite matrix (2026-07-23)

- Model: dual-masked LoRA (pre-rescue)
- Lang method: 1 fresh baseline probe + 4 draws from BG/color pool (mean-of-5)

### Suite × condition (origin / lang / bg / color)

| Suite | origin | lang-l1 | lang-l2 | bg-0 | bg-1 | bg-2 | color-0 | color-1 | pooled |
|-------|--------|---------|---------|------|------|------|---------|---------|--------|
| goal | 4/30 (13%) | 10% | 12% | 5/30 (17%) | 6/30 (20%) | 5/30 (17%) | 0/30 (0%) | 1/30 (3%) | 21/180 (12%) |
| object | 12/30 (40%) | 30% | 38% | 10/30 (33%) | 12/30 (40%) | 11/30 (37%) | 9/30 (30%) | 9/30 (30%) | 63/180 (35%) |
| spatial | 6/30 (20%) | 14% | 16% | 5/30 (17%) | 3/30 (10%) | 2/30 (7%) | 2/30 (7%) | 1/30 (3%) | 19/180 (11%) |
| study_scene4 | 5/12 (42%) | 20% | 30% | 4/12 (33%) | 4/12 (33%) | 3/12 (25%) | 3/12 (25%) | 5/12 (42%) | 24/72 (33%) |

### Pre-rescue language-shift (mean-of-5) per task

| Suite | Task | new | lang-l1 | lang-l2 |
|-------|------|-----|---------|---------|
| goal | open the middle drawer of the cabinet | 0 | 0% | 0% |
| goal | open the top drawer and put the bowl inside | 0 | 0% | 0% |
| goal | push the plate to the front of the stove | 0 | 0% | 0% |
| goal | put the bowl on the plate | 1 | 40% | 80% |
| goal | put the bowl on the stove | 0 | 40% | 40% |
| goal | put the bowl on top of the cabinet | 0 | 20% | 0% |
| goal | put the cream cheese in the bowl | 0 | 0% | 0% |
| goal | put the wine bottle on the rack | 0 | 0% | 0% |
| goal | put the wine bottle on top of the cabinet | 0 | 0% | 0% |
| goal | turn on the stove | 0 | 0% | 0% |
| object | pick up the alphabet soup and place it in the basket | 0 | 0% | 0% |
| object | pick up the bbq sauce and place it in the basket | 1 | 60% | 60% |
| object | pick up the butter and place it in the basket | 1 | 80% | 100% |
| object | pick up the chocolate pudding and place it in the basket | 0 | 0% | 0% |
| object | pick up the cream cheese and place it in the basket | 0 | 0% | 0% |
| object | pick up the ketchup and place it in the basket | 0 | 0% | 20% |
| object | pick up the milk and place it in the basket | 0 | 0% | 0% |
| object | pick up the orange juice and place it in the basket | 1 | 60% | 80% |
| object | pick up the salad dressing and place it in the basket | 1 | 100% | 100% |
| object | pick up the tomato sauce and place it in the basket | 0 | 0% | 20% |
| spatial | pick up the black bowl between the plate and the ramekin ... | 1 | 80% | 60% |
| spatial | pick up the black bowl from table center and place it on ... | 0 | 0% | 0% |
| spatial | pick up the black bowl in the top drawer of the wooden ca... | 1 | 20% | 40% |
| spatial | pick up the black bowl next to the cookie box and place i... | 1 | 20% | 20% |
| spatial | pick up the black bowl next to the plate and place it on ... | 0 | 0% | 0% |
| spatial | pick up the black bowl next to the ramekin and place it o... | 0 | 0% | 0% |
| spatial | pick up the black bowl on the cookie box and place it on ... | 1 | 20% | 40% |
| spatial | pick up the black bowl on the ramekin and place it on the... | 0 | 0% | 0% |
| spatial | pick up the black bowl on the stove and place it on the p... | 0 | 0% | 0% |
| spatial | pick up the black bowl on the wooden cabinet and place it... | 0 | 0% | 0% |
| study_scene4 | pick up the book in the middle and place it on the cabine... | 0 | 20% | 0% |
| study_scene4 | pick up the book on the left and place it on top of the s... | 0 | 0% | 0% |
| study_scene4 | pick up the book on the right and place it on the cabinet... | 1 | 60% | 100% |
| study_scene4 | pick up the book on the right and place it under the cabi... | 0 | 0% | 20% |

## 2. reported_success = 1 (mostly working, ≥~50%)

- **[goal]** put the bowl on the plate: 10/18 (56%)
- **[goal]** put the bowl on the stove: 10/18 (56%)
- **[object]** pick up the bbq sauce and place it in the basket: 11/18 (61%)
- **[object]** pick up the butter and place it in the basket: 15/18 (83%)
- **[object]** pick up the orange juice and place it in the basket: 13/18 (72%)
- **[object]** pick up the salad dressing and place it in the basket: 18/18 (100%)
- **[spatial]** pick up the black bowl between the plate and the ramekin and place it on the plate: 10/18 (56%)
- **[study_scene4]** pick up the book on the right and place it on the cabinet shelf: 15/18 (83%)

## 3. Low-SR tasks (<15%) — rescue status

| Suite | Task | Pool SR | Rescue |
|-------|------|---------|--------|
| goal | open the middle drawer of the cabinet | 0% | goal-low / push-rescue (see §8) |
| goal | open the top drawer and put the bowl inside | 0% | goal-low / push-rescue (see §8) |
| goal | push the plate to the front of the stove | 0% | goal-low / push-rescue (see §8) |
| goal | put the bowl on top of the cabinet | 6% | goal-low / push-rescue (see §8) |
| goal | put the cream cheese in the bowl | 0% | goal-low / push-rescue (see §8) |
| goal | put the wine bottle on the rack | 0% | goal-low / push-rescue (see §8) |
| goal | put the wine bottle on top of the cabinet | 0% | goal-low / push-rescue (see §8) |
| goal | turn on the stove | 0% | goal-low / push-rescue (see §8) |
| object | pick up the alphabet soup and place it in the basket | 0% | object-rescue +20k GPU2 |
| object | pick up the chocolate pudding and place it in the basket | 0% | object-rescue +20k GPU2 |
| object | pick up the cream cheese and place it in the basket | 11% | object-rescue +20k GPU2 |
| object | pick up the ketchup and place it in the basket | 11% | object-rescue +20k GPU2 |
| object | pick up the milk and place it in the basket | 0% | object-rescue +20k GPU2 |
| object | pick up the tomato sauce and place it in the basket | 11% | object-rescue +20k GPU2 |
| spatial | pick up the black bowl from table center and place it on the plate | 0% | spatial-rescue +20k GPU4 |
| spatial | pick up the black bowl next to the cookie box and place it on the plate | 11% | spatial-rescue +20k GPU4 |
| spatial | pick up the black bowl next to the plate and place it on the plate | 0% | spatial-rescue +20k GPU4 |
| spatial | pick up the black bowl next to the ramekin and place it on the plate | 0% | spatial-rescue +20k GPU4 |
| spatial | pick up the black bowl on the ramekin and place it on the plate | 0% | spatial-rescue +20k GPU4 |
| spatial | pick up the black bowl on the stove and place it on the plate | 0% | spatial-rescue +20k GPU4 |
| spatial | pick up the black bowl on the wooden cabinet and place it on the plate | 6% | spatial-rescue +20k GPU4 |
| study_scene4 | pick up the book on the left and place it on top of the shelf | 0% | ongoing study_scene4 train →297k |

## 4. Language-shift (ours mean-of-5)

Method: **1 new baseline trial / task** + **4 random draws** from existing BG/color trials → mean. l1/l2 use different RNG seeds.
Note: masked-language input unchanged for ours; new trial ≈ another origin sample.

| Suite | lang-l1 | lang-l2 |
|-------|---------|---------|
| goal | 10% | 12% |
| object | 30% | 38% |
| spatial | 14% | 16% |
| study_scene4 | 20% | 30% |

## 5. Per-task × visual condition

| Suite | Task | origin | bg-0 | bg-1 | bg-2 | color-0 | color-1 | pooled |
|-------|------|------|------|------|------|------|------|--------|
| goal | open the middle drawer of the cabinet | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| goal | open the top drawer and put the bowl inside | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| goal | push the plate to the front of the stove | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| goal | put the bowl on the plate | 2/3 | 2/3 | 3/3 | 2/3 | 0/3 | 1/3 | 10/18 |
| goal | put the bowl on the stove | 2/3 | 3/3 | 3/3 | 2/3 | 0/3 | 0/3 | 10/18 |
| goal | put the bowl on top of the cabinet | 0/3 | 0/3 | 0/3 | 1/3 | 0/3 | 0/3 | 1/18 |
| goal | put the cream cheese in the bowl | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| goal | put the wine bottle on the rack | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| goal | put the wine bottle on top of the cabinet | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| goal | turn on the stove | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| object | pick up the alphabet soup and place it in the basket | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| object | pick up the bbq sauce and place it in the basket | 2/3 | 2/3 | 2/3 | 2/3 | 2/3 | 1/3 | 11/18 |
| object | pick up the butter and place it in the basket | 3/3 | 3/3 | 3/3 | 2/3 | 2/3 | 2/3 | 15/18 |
| object | pick up the chocolate pudding and place it in the basket | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| object | pick up the cream cheese and place it in the basket | 0/3 | 0/3 | 0/3 | 0/3 | 1/3 | 1/3 | 2/18 |
| object | pick up the ketchup and place it in the basket | 2/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 2/18 |
| object | pick up the milk and place it in the basket | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| object | pick up the orange juice and place it in the basket | 2/3 | 2/3 | 3/3 | 3/3 | 1/3 | 2/3 | 13/18 |
| object | pick up the salad dressing and place it in the basket | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 18/18 |
| object | pick up the tomato sauce and place it in the basket | 0/3 | 0/3 | 1/3 | 1/3 | 0/3 | 0/3 | 2/18 |
| spatial | pick up the black bowl between the plate and the ramekin and place ... | 3/3 | 2/3 | 2/3 | 2/3 | 1/3 | 0/3 | 10/18 |
| spatial | pick up the black bowl from table center and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| spatial | pick up the black bowl in the top drawer of the wooden cabinet and ... | 1/3 | 1/3 | 1/3 | 0/3 | 0/3 | 0/3 | 3/18 |
| spatial | pick up the black bowl next to the cookie box and place it on the p... | 1/3 | 0/3 | 0/3 | 0/3 | 1/3 | 0/3 | 2/18 |
| spatial | pick up the black bowl next to the plate and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| spatial | pick up the black bowl next to the ramekin and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| spatial | pick up the black bowl on the cookie box and place it on the plate | 1/3 | 1/3 | 0/3 | 0/3 | 0/3 | 1/3 | 3/18 |
| spatial | pick up the black bowl on the ramekin and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| spatial | pick up the black bowl on the stove and place it on the plate | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| spatial | pick up the black bowl on the wooden cabinet and place it on the plate | 0/3 | 1/3 | 0/3 | 0/3 | 0/3 | 0/3 | 1/18 |
| study_scene4 | pick up the book in the middle and place it on the cabinet shelf | 1/3 | 0/3 | 0/3 | 0/3 | 1/3 | 1/3 | 3/18 |
| study_scene4 | pick up the book on the left and place it on top of the shelf | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/18 |
| study_scene4 | pick up the book on the right and place it on the cabinet shelf | 3/3 | 3/3 | 2/3 | 2/3 | 2/3 | 3/3 | 15/18 |
| study_scene4 | pick up the book on the right and place it under the cabinet shelf | 1/3 | 1/3 | 2/3 | 1/3 | 0/3 | 1/3 | 6/18 |

## 6. Early goal matrix

(see previous RESULTS snapshot / logs if needed)

## 7. Hidden / visual / lang（汇总）

**主表与定义见 → [`SHIFT_AND_HIDDEN.md`](./SHIFT_AND_HIDDEN.md)**（visual · lang · hidden）。

### Post-rescue hidden similarity（vlm_prefix_l18）

| Suite | lang | img full | img non-black | img R∪G |
|-------|------|----------|---------------|---------|
| goal | 0.9998 | 0.9766 | 0.7213 | 0.6409 |
| object | 0.9998 | 0.9795 | 0.8359 | 0.8044 |
| spatial | 0.9997 | 0.9757 | 0.8564 | 0.7982 |
| study_scene4 | 0.9998 | 0.9714 | 0.8151 | 0.7745 |

Action 干预见 `SHIFT_AND_HIDDEN.md` §3b。 JSON: `grounding_claim_openvla_post_rescue.json`。


## 8. Training / rescue timeline (live)

| Job | GPU | Range | Filter | Status |
|-----|-----|-------|--------|--------|
| goal push-rescue | 5,6 | →671.5k | push + put bowl | running / check |
| goal low-rescue | 5,6 | →691.5k | all low goal tasks | queued after push |
| object-rescue | 2 | →246.5k | 6 low object tasks | running / check |
| spatial-rescue | 4 | →638k | 9 low spatial tasks | running / check |
| study_scene4 | 3 | →297k | all 4 book tasks | running |

## 9. Baselines (OFT / π)

Not run yet in this matrix (disk / queue).

---

## 10. Mask-model RL LoRA — `libero_goal` color/bowl OOD (2026-08-25)

**Backbone (SFT, mask model):**  
`/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/openvla_adapters/openvla-7b+dual_masked_goal+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_goal_oft_lr`

**Method:** continuous OpenVLA-OFT RL (`run_rl_lora_color_quick.py`); **action_head-only** (vision LoRA frozen); only update groups with ≥1 success; BC on **greedy** success trajs when available; rollback on collapse; **LoRA not merged** into SFT.

**Eval:** greedy, 4 trials × 2 color variants (8 episodes); baseline SR from prior matrix / §0 (not re-run).

**Launcher:** `tools/run_rl_lora_color_quick.sh`, batch `tools/run_all_goal_rl_stable.sh`

**Run root:**  
`/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/rl_lora_color_quick/`

### 10.1 Results (recommended run per task)

| id | Task | OOD | Baseline (prior) | RL final SR | Δ | Verdict |
|----|------|-----|------------------|-------------|---|---------|
| 0 | open the middle drawer of the cabinet | plate **colors** | color mid-high | **87.5%** (7/8) | ↑ | ✅ |
| 1 | put the bowl on the stove | **bowl** | ~40% | **100%** (8/8) | ↑↑ | ✅ |
| 2 | put the wine bottle on top of the cabinet | plate **colors** | ~83% | **87.5%** (7/8) | ~ | ✅ (fixed explore→greedy gap) |
| 3 | open the top drawer and put the bowl inside | **bowl** | color **0%** | **0%** | — | ❌ no train signal (ckpt removed) |
| 4 | put the bowl on top of the cabinet | **bowl** | mid | **87.5%** (7/8) | ↑ | ✅ |
| 5 | push the plate to the front of the stove | plate **colors** | ~33% (1/3+1/3) | **37.5%** (3/8) | ~ | mild |
| 6 | put the cream cheese in the bowl | plate **colors** | ~17% | **25%** (2/8) | ~ | mild (`stable_t6`) |
| 7 | turn on the stove | plate **colors** | high (~100%) | **100%** (8/8) | maintain | ✅ |
| 8 | put the bowl on the plate | **bowl** | **40%** (matrix) | **87.5%** (7/8) | **+47.5pp** | ✅ **primary** |
| 9 | put the wine bottle on the rack | plate **colors** | high (~100%) | **100%** (8/8) | maintain | ✅ |

**Headline cases for “mid SR → RL higher”:** **#8** (40→87.5%), **#4** (→87.5%), **#1** (→100%), **#0** (→87.5%).  
**#2** recovered after greedy-BC recipe (0→87.5%). **#5/#6** weak; **#3** unsuitable.

### 10.2 RL checkpoints (use `best/` for eval)

Each run dir: `{run_root}/{run_name}/best/` contains unmerged RL adapter + heads.  
Load: SFT checkpoint + `best/lora_adapter` + `best/action_head--rl_checkpoint.pt` (+ `best/proprio_projector--rl_checkpoint.pt`).

| id | Run name | Final SR | Best ckpt directory |
|----|----------|----------|---------------------|
| 0 | `libero_goal_task0_allg_t0` | 87.5% | `.../rl_lora_color_quick/libero_goal_task0_allg_t0/best/` |
| 1 | `libero_goal_task1_allg_t1` | 100% | `.../rl_lora_color_quick/libero_goal_task1_allg_t1/best/` |
| 2 | `libero_goal_task2_allg_t2` | 87.5% | `.../rl_lora_color_quick/libero_goal_task2_allg_t2/best/` |
| 3 | — | 0% | *(deleted — no signal)* |
| 4 | `libero_goal_task4_stable_t4` | 87.5% | `.../rl_lora_color_quick/libero_goal_task4_stable_t4/best/` |
| 5 | `libero_goal_task5_mask_plate5_stable` | 37.5% | `.../rl_lora_color_quick/libero_goal_task5_mask_plate5_stable/best/` |
| 6 | `libero_goal_task6_stable_t6` | 25% | `.../rl_lora_color_quick/libero_goal_task6_stable_t6/best/` |
| 7 | `libero_goal_task7_allg_t7` | 100% | `.../rl_lora_color_quick/libero_goal_task7_allg_t7/best/` |
| 8 | `libero_goal_task8_mask_bowl8_stable` | 87.5% | `.../rl_lora_color_quick/libero_goal_task8_mask_bowl8_stable/best/` |
| 9 | `libero_goal_task9_allg_t9` | 100% | `.../rl_lora_color_quick/libero_goal_task9_allg_t9/best/` |

Full prefix: `/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/rl_lora_color_quick/`

**Per-run artifacts under `best/` (and mirrored at run root after last iter):**

| File | Role |
|------|------|
| `lora_adapter/` | Unmerged RL vision LoRA (rank 8; often empty when `train_vision_lora=False`) |
| `action_head--rl_checkpoint.pt` | Trained continuous action head (~289M) |
| `proprio_projector--rl_checkpoint.pt` | Proprio projector |
| `rl_meta.json` | Best iter, train_sr, config snapshot |
| `SUMMARY.json` | Final greedy SR (run root) |
| `train_history.json` | Per-iter metrics, rollbacks (run root) |

**Example — task 8 (primary):**

```
/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/runs/rl_lora_color_quick/libero_goal_task8_mask_bowl8_stable/best/
├── lora_adapter/
├── action_head--rl_checkpoint.pt
├── proprio_projector--rl_checkpoint.pt
└── rl_meta.json
```

**Superseded / collapsed runs (deleted 2026-08-26):**  
`task8_mask_bowl8_color`, `task5_mask_plate5_color`, `task1_stable_t1`, `task2_stable_t2`, `task3_allg_t3`, `task6_allg_t6`, empty `task3_stable_t3`, `spatial_task2_stable_s2`.

**Logs:** `openvla-oft/logs/rl_lora_color_goal_task*_*.log`

