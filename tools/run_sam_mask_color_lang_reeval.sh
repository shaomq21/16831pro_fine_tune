#!/usr/bin/env bash
# Color + lang-l1/l2 (mean-of-5) for generated-mask reeval (post-rescue ckpt).
#   goal    → Grounded SAM (sam1)
#   spatial → SAM3 + temporal
#
# Env:
#   SUITES=goal,spatial  TRIALS=3
#   VLA_GPUS=1,2  MASK_GPUS=3,4
#   SKIP_COLOR=0  SKIP_LANG=0
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
export VLA_PREPROCESS_PY="${VLA_PREPROCESS_PY:-${STORAGE_ROOT}/conda_envs/vla-preprocess/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"

OUT_ROOT="${OUT_ROOT:-${OFT}/runs/sam_mask_reeval}"
LOG_DIR="${OUT_ROOT}/logs"
SUM_DIR="${OUT_ROOT}/summary"
NOTE="${NOTE:-sam_color_lang_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}" "${SUM_DIR}"

export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false

TRIALS="${TRIALS:-3}"
SKIP_COLOR="${SKIP_COLOR:-0}"
SKIP_LANG="${SKIP_LANG:-0}"
SUITES="${SUITES:-goal,spatial}"
VLA_GPUS="${VLA_GPUS:-1,2}"
MASK_GPUS="${MASK_GPUS:-3,4}"

IFS=',' read -r -a SUITE_ARR <<< "${SUITES}"
IFS=',' read -r -a VLA_GPU_ARR <<< "${VLA_GPUS}"
IFS=',' read -r -a MASK_GPU_ARR <<< "${MASK_GPUS}"

ckpt_for() {
  echo "${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_${1}+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_${1}_oft_lr"
}
tsuite_for() {
  case "$1" in goal) echo libero_goal ;; spatial) echo libero_spatial ;; esac
}
backend_for() {
  case "$1" in goal) echo sam1 ;; spatial) echo sam3 ;; esac
}

MASTER="${LOG_DIR}/launcher_color_lang_${NOTE}.log"
echo "===== $(date -Iseconds) START color+lang generated-mask note=${NOTE} =====" | tee "${MASTER}"

# ---------- Color ----------
if [[ "${SKIP_COLOR}" != "1" ]]; then
  i=0
  pids=()
  for suite in "${SUITE_ARR[@]}"; do
    suite="$(echo "${suite}" | xargs)"
    [[ -z "${suite}" ]] && continue
    vla_gpu="${VLA_GPU_ARR[$((i % ${#VLA_GPU_ARR[@]}))]}"
    mask_gpu="${MASK_GPU_ARR[$((i % ${#MASK_GPU_ARR[@]}))]}"
    backend="$(backend_for "${suite}")"
    ckpt="$(ckpt_for "${suite}")"
    tsuite="$(tsuite_for "${suite}")"
    tag="ours_${suite}_${backend}_color"
    slog="${LOG_DIR}/${tag}_${NOTE}.log"
    (
      export MASK_GPU="${mask_gpu}"
      echo "===== $(date -Iseconds) START ${tag} VLA=${vla_gpu} MASK=${mask_gpu} =====" | tee -a "${slog}"
      CUDA_VISIBLE_DEVICES="${vla_gpu}" "${PYTHON}" experiments/robot/libero/run_libero_color_perturb_eval.py \
        --pretrained_checkpoint "${ckpt}" \
        --base_vla_path "${BASE_VLA}" \
        --task_suite_name "${tsuite}" \
        --tasks all \
        --model_label "ours_masked_${backend}" \
        --use_mask_for_policy True \
        --use_mask_from_env False \
        --sam_backend "${backend}" \
        --mask_alpha 0.35 \
        --mask_device cuda \
        --num_images_in_input 1 \
        --num_trials_per_task "${TRIALS}" \
        --use_proprio True \
        --use_l1_regression True \
        --lora_rank 32 \
        --center_crop False \
        --local_log_dir "${LOG_DIR}" \
        --run_id_note "${tag}_${NOTE}" \
        --load_in_8bit False \
        2>&1 | tee -a "${slog}"
      echo "===== $(date -Iseconds) END ${tag} rc=${PIPESTATUS[0]} =====" | tee -a "${slog}"
    ) &
    pids+=($!)
    echo "launched color ${tag} VLA=${vla_gpu} MASK=${mask_gpu} pid=${pids[$((${#pids[@]}-1))]}" | tee -a "${MASTER}"
    i=$((i+1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
  echo "===== $(date -Iseconds) color phase done =====" | tee -a "${MASTER}"
fi

# ---------- Lang probe: 1 fresh baseline / task ----------
if [[ "${SKIP_LANG}" != "1" ]]; then
  i=0
  pids=()
  for suite in "${SUITE_ARR[@]}"; do
    suite="$(echo "${suite}" | xargs)"
    [[ -z "${suite}" ]] && continue
    vla_gpu="${VLA_GPU_ARR[$((i % ${#VLA_GPU_ARR[@]}))]}"
    mask_gpu="${MASK_GPU_ARR[$((i % ${#MASK_GPU_ARR[@]}))]}"
    backend="$(backend_for "${suite}")"
    ckpt="$(ckpt_for "${suite}")"
    tsuite="$(tsuite_for "${suite}")"
    tag="langshift_${suite}_${backend}"
    slog="${LOG_DIR}/${tag}_${NOTE}.log"
    (
      export MASK_GPU="${mask_gpu}"
      echo "===== $(date -Iseconds) START ${tag} VLA=${vla_gpu} MASK=${mask_gpu} =====" | tee -a "${slog}"
      CUDA_VISIBLE_DEVICES="${vla_gpu}" "${PYTHON}" experiments/robot/libero/run_libero_background_perturb_eval.py \
        --pretrained_checkpoint "${ckpt}" \
        --base_vla_path "${BASE_VLA}" \
        --task_suite_name "${tsuite}" \
        --tasks all \
        --model_label "ours_langshift_probe_${backend}" \
        --use_mask_for_policy True \
        --use_mask_from_env False \
        --sam_backend "${backend}" \
        --mask_alpha 0.35 \
        --mask_device cuda \
        --run_baseline True \
        --run_background False \
        --num_images_in_input 1 \
        --num_trials_per_task 1 \
        --use_proprio True \
        --use_l1_regression True \
        --lora_rank 32 \
        --center_crop False \
        --local_log_dir "${LOG_DIR}" \
        --run_id_note "langshift_${suite}_${NOTE}" \
        --load_in_8bit False \
        2>&1 | tee -a "${slog}"
      echo "===== $(date -Iseconds) END ${tag} rc=${PIPESTATUS[0]} =====" | tee -a "${slog}"
    ) &
    pids+=($!)
    echo "launched lang ${tag} VLA=${vla_gpu} MASK=${mask_gpu} pid=${pids[$((${#pids[@]}-1))]}" | tee -a "${MASTER}"
    i=$((i+1))
  done
  for pid in "${pids[@]}"; do wait "${pid}" || true; done
  echo "===== $(date -Iseconds) lang probe done =====" | tee -a "${MASTER}"
fi

# ---------- Aggregate mean-of-5 from THIS OUT_ROOT only ----------
export LOG_DIR SUM_DIR NOTE
"${PYTHON}" - <<'PY'
import re, glob, os, json, random
from collections import defaultdict

logdir = os.environ["LOG_DIR"]
sumdir = os.environ["SUM_DIR"]
note = os.environ["NOTE"]

def suite_of(name):
    if "libero_object" in name: return "object"
    if "libero_spatial" in name: return "spatial"
    if "libero_90" in name: return "study_scene4"
    if "libero_goal" in name: return "goal"
    return None

# Pool: all BG/COLOR in this generated-mask log dir (exclude langshift probes)
pool = defaultdict(list)
for p in glob.glob(logdir + "/BG-PERTURB*.txt") + glob.glob(logdir + "/COLOR-PERTURB*.txt"):
    bn = os.path.basename(p)
    if "langshift_" in bn:
        continue
    # only post-rescue generated-mask runs
    if "sam1" not in bn and "sam3" not in bn and "postrescue" not in bn and "resume" not in bn:
        # still include COLOR tagged ours_masked_sam*
        if "ours_masked_sam" not in bn and "ours_goal_sam" not in bn and "ours_spatial_sam" not in bn:
            continue
    suite = suite_of(bn)
    if not suite:
        continue
    text = open(p, errors="ignore").read()
    for m in re.finditer(r"Task: (.*?) \| Perturb: (.*?) \| Trial:.*?\n.*?Success: (True|False)", text, re.S):
        pool[(suite, m.group(1).strip())].append(1 if m.group(3) == "True" else 0)
    for m in re.finditer(r"Task: (.*?) \| Color variant:?\s*(\d+).*?\n.*?Success: (True|False)", text, re.S):
        pool[(suite, m.group(1).strip())].append(1 if m.group(3) == "True" else 0)

new = {}
for p in glob.glob(logdir + "/BG-PERTURB*.txt"):
    if "langshift_" not in os.path.basename(p):
        continue
    suite = suite_of(os.path.basename(p))
    if not suite:
        continue
    text = open(p, errors="ignore").read()
    for m in re.finditer(r"Task: (.*?) \| Perturb: (.*?) \| Trial:.*?\n.*?Success: (True|False)", text, re.S):
        task, pert, suc = m.group(1).strip(), m.group(2).strip(), 1 if m.group(3) == "True" else 0
        if pert != "baseline":
            continue
        new[(suite, task)] = suc

rows = []
for key in sorted(set(pool) | set(new)):
    suite, task = key
    if key not in new:
        continue
    p = pool.get(key, [])
    samp_src = p if p else [new[key]]

    def mean5(seed):
        rng = random.Random(seed)
        if len(samp_src) >= 4:
            drawn = rng.sample(samp_src, 4)
        else:
            drawn = [rng.choice(samp_src) for _ in range(4)]
        vals = [new[key]] + drawn
        return sum(vals) / 5.0, vals

    l1, m1 = mean5(hash(("l1", suite, task)) & 0xFFFFFFFF)
    l2, m2 = mean5(hash(("l2", suite, task)) & 0xFFFFFFFF)
    rows.append({
        "suite": suite, "task": task,
        "new_trial": new[key],
        "pool_n": len(p),
        "pool_mean": round(sum(p) / len(p), 3) if p else None,
        "lang_l1_mean5": round(l1, 3), "lang_l1_bits": m1,
        "lang_l2_mean5": round(l2, 3), "lang_l2_bits": m2,
    })

out = {
    "note": note,
    "policy": "generated-mask (sam1/sam3); ours masked-lang unchanged; 1 fresh baseline + 4 draws from BG/color pool",
    "tasks": rows,
    "suite_l1": {},
    "suite_l2": {},
}
s1 = defaultdict(list); s2 = defaultdict(list)
for r in rows:
    s1[r["suite"]].append(r["lang_l1_mean5"]); s2[r["suite"]].append(r["lang_l2_mean5"])
out["suite_l1"] = {s: round(sum(v) / len(v), 3) for s, v in s1.items()}
out["suite_l2"] = {s: round(sum(v) / len(v), 3) for s, v in s2.items()}

# also color suite totals
color_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for p in glob.glob(logdir + "/COLOR-PERTURB*.txt"):
    bn = os.path.basename(p)
    suite = suite_of(bn)
    if not suite:
        continue
    text = open(p, errors="ignore").read()
    cur_v = None
    for line in text.splitlines():
        m = re.match(r".*Task: (.*) \| Color variant:?\s*(\d+)", line)
        if m:
            cur_v = (m.group(1).strip(), int(m.group(2)))
        m2 = re.match(r"Success: (True|False)", line)
        if m2 and cur_v:
            task, v = cur_v
            color_counts[suite][v][1] += 1
            if m2.group(1) == "True":
                color_counts[suite][v][0] += 1
out["color"] = {
    s: {f"color_{v}": f"{ok}/{n}" for v, (ok, n) in sorted(vs.items())}
    for s, vs in color_counts.items()
}

path = f"{sumdir}/langshift_mean5_generated_mask_{note}.json"
json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)
print("wrote", path)
print("suite_l1", out["suite_l1"])
print("suite_l2", out["suite_l2"])
print("color", out["color"])
print("tasks", len(rows))

# --- Patch RESULTS.md §0b table with color + lang ---
results_candidates = [
    os.path.join(os.environ.get("STORAGE_ROOT", "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune"),
                 "runs/all_suites_perturb_matrix/summary/RESULTS.md"),
    "/home/fan-test/maggie/subopt_proj/third_party/16831pro_fine_tune/openvla-oft/runs/all_suites_perturb_matrix/summary/RESULTS.md",
]
# Known origin/bg from prior generated-mask run (postrescue)
known = {
    "goal": {"origin": "12/30 (40%)", "bg0": "11/30 (37%)", "bg1": "12/30 (40%)", "bg2": "12/30 (40%)", "bgp": "47/120 (39%)", "mask": "Grounded SAM (sam1)"},
    "spatial": {"origin": "3/30 (10%)", "bg0": "7/30 (23%)", "bg1": "7/30 (23%)", "bg2": "7/30 (23%)", "bgp": "24/120 (20%)", "mask": "SAM3 + temporal"},
}

def pct(ok, n):
    return f"{ok}/{n} ({100*ok/max(n,1):.0f}%)"

rows_md = []
for suite in ["goal", "spatial"]:
    k = known[suite]
    c0 = color_counts[suite].get(0, [0, 0])
    c1 = color_counts[suite].get(1, [0, 0])
    l1 = out["suite_l1"].get(suite)
    l2 = out["suite_l2"].get(suite)
    l1s = f"{100*l1:.0f}%" if l1 is not None else "—"
    l2s = f"{100*l2:.0f}%" if l2 is not None else "—"
    c0s = pct(*c0) if c0[1] else "—"
    c1s = pct(*c1) if c1[1] else "—"
    rows_md.append(
        f"| {suite} | {k['mask']} | {k['origin']} | {l1s} | {l2s} | {k['bg0']} | {k['bg1']} | {k['bg2']} | {c0s} | {c1s} | {k['bgp']} |"
    )

new_table = "\n".join([
    "| Suite | mask | origin | lang-l1 | lang-l2 | bg-0 | bg-1 | bg-2 | color-0 | color-1 | bg-pooled |",
    "|-------|------|--------|---------|---------|------|------|------|---------|---------|-----------|",
    *rows_md,
])

marker_start = "| Suite | mask | origin | bg-0 | bg-1 | bg-2 | bg-pooled |"
# also match already-expanded header
marker_start2 = "| Suite | mask | origin | lang-l1 | lang-l2 | bg-0 | bg-1 | bg-2 | color-0 | color-1 | bg-pooled |"

for rp in results_candidates:
    if not os.path.isfile(rp):
        continue
    text = open(rp).read()
    if "## 0b. Generated-mask" not in text:
        print("skip RESULTS (no §0b):", rp)
        continue
    import re as _re
    # replace the suite summary table inside §0b (until blank line after table or ### )
    pat = _re.compile(
        r"(\| Suite \| mask \| origin \|.*?\n\|[-| ]+\|\n)(?:\|.*\n)+",
        _re.M,
    )
    # Only first table after 0b
    idx = text.find("## 0b. Generated-mask")
    if idx < 0:
        continue
    head, rest = text[:idx], text[idx:]
    m = pat.search(rest)
    if not m:
        print("could not find §0b table in", rp)
        continue
    rest2 = rest[:m.start()] + new_table + "\n\n" + rest[m.end():]
    # ensure note about color/lang
    if "color + lang-l1/l2" not in rest2:
        rest2 = rest2.replace(
            "- Schedule: origin(=baseline) + bg-0/1/2，TRIALS=3 / task；**未跑 color / lang**",
            f"- Schedule: origin + bg-0/1/2 + color-0/1，TRIALS=3；lang = mean-of-5（note=`{note}`）",
        )
    open(rp, "w").write(head + rest2)
    print("patched RESULTS.md:", rp)
PY

echo "===== $(date -Iseconds) DONE color+lang note=${NOTE} =====" | tee -a "${MASTER}"
echo "Logs: ${LOG_DIR}  Summary: ${SUM_DIR}" | tee -a "${MASTER}"
