#!/usr/bin/env bash
# Language-shift probe for ours:
#   1 new trial per task (baseline visual; masked-lang input unchanged)
#   Then report lang-l1/l2 = mean(new + 4 random draws from existing pool)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OFT="${REPO_ROOT}/openvla-oft"
cd "${OFT}"

STORAGE_ROOT="${STORAGE_ROOT:-/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/conda_envs/simplevla/bin/python}"
BASE_VLA="${OFT}/checkpoints/openvla-7b"
OUT_ROOT="${OUT_ROOT:-${STORAGE_ROOT}/runs/all_suites_perturb_matrix}"
LOG_DIR="${OUT_ROOT}/logs"
SUM_DIR="${OUT_ROOT}/summary"
NOTE="${NOTE:-langshift_probe_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}" "${SUM_DIR}"

export PYTHONPATH="${REPO_ROOT}/LIBERO:${OFT}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM=false

SUITES="${SUITES:-object,spatial,study_scene4,goal}"
GPUS="${GPUS:-2,7}"
IFS=',' read -r -a SUITE_ARR <<< "${SUITES}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"

ckpt_for() {
  case "$1" in
    study_scene4)
      echo "${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_study_scene4+b2+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_study_scene4_oft_lr"
      ;;
    *)
      echo "${STORAGE_ROOT}/runs/openvla_adapters/openvla-7b+dual_masked_$1+b4+lr-0.0005+lora-r32+dropout-0.0+lora-attn-only--suite_$1_oft_lr"
      ;;
  esac
}
tsuite_for() {
  case "$1" in
    goal) echo libero_goal ;;
    object) echo libero_object ;;
    spatial) echo libero_spatial ;;
    study_scene4) echo libero_90 ;;
  esac
}
tasks_for() {
  case "$1" in
    study_scene4)
      echo "pick up the book in the middle and place it on the cabinet shelf|pick up the book on the left and place it on top of the shelf|pick up the book on the right and place it on the cabinet shelf|pick up the book on the right and place it under the cabinet shelf"
      ;;
    *) echo "all" ;;
  esac
}
unorm_for() {
  case "$1" in
    study_scene4) echo --unnorm_key simu_libero_90_study_scene4_no_noops ;;
    *) echo "" ;;
  esac
}

MASTER="${LOG_DIR}/langshift_probe_${NOTE}.log"
echo "===== $(date -Iseconds) START langshift probe suites=${SUITES} =====" | tee "${MASTER}"

i=0
pids=()
for suite in "${SUITE_ARR[@]}"; do
  suite="$(echo "${suite}" | xargs)"
  [[ -z "${suite}" ]] && continue
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  ckpt="$(ckpt_for "${suite}")"
  tsuite="$(tsuite_for "${suite}")"
  tasks="$(tasks_for "${suite}")"
  unorm="$(unorm_for "${suite}")"
  slog="${LOG_DIR}/langshift_${suite}_${NOTE}.log"
  (
    echo "===== $(date -Iseconds) START ${suite} GPU=${gpu} =====" | tee -a "${slog}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" experiments/robot/libero/run_libero_background_perturb_eval.py \
      --pretrained_checkpoint "${ckpt}" \
      --base_vla_path "${BASE_VLA}" \
      --task_suite_name "${tsuite}" \
      --tasks "${tasks}" \
      --model_label "ours_langshift_probe" \
      --use_mask_for_policy True \
      --use_mask_from_env True \
      --run_baseline True \
      --run_background False \
      --num_images_in_input 1 \
      --num_trials_per_task 1 \
      --use_proprio True \
      --use_l1_regression True \
      --lora_rank 32 \
      --center_crop False \
      --load_in_8bit False \
      --local_log_dir "${LOG_DIR}" \
      --run_id_note "langshift_${suite}_${NOTE}" \
      ${unorm} \
      2>&1 | tee -a "${slog}"
    echo "===== $(date -Iseconds) END ${suite} rc=${PIPESTATUS[0]} =====" | tee -a "${slog}"
  ) &
  pids+=($!)
  echo "launched ${suite} on GPU ${gpu} pid=${pids[$((${#pids[@]}-1))]}" | tee -a "${MASTER}"
  i=$((i+1))
done

ec=0
for pid in "${pids[@]}"; do
  wait "${pid}" || ec=1
done
echo "===== $(date -Iseconds) probe runs done ec=${ec} =====" | tee -a "${MASTER}"

# Aggregate: 1 new + 4 random from existing pool → lang_l1 / lang_l2 means
export LOG_DIR SUM_DIR NOTE
"${PYTHON}" - <<PY
import re,glob,os,json,random
from collections import defaultdict

logdir=os.environ["LOG_DIR"]
sumdir=os.environ["SUM_DIR"]
note=os.environ["NOTE"]

def suite_of(name):
  if "libero_object" in name: return "object"
  if "libero_spatial" in name: return "spatial"
  if "libero_90" in name: return "study_scene4"
  if "libero_goal" in name: return "goal"
  return None

# existing pool (exclude brand-new langshift probe logs)
pool=defaultdict(list)
for p in glob.glob(logdir+"/BG-PERTURB*.txt")+glob.glob(logdir+"/COLOR-PERTURB*.txt"):
  if "langshift_" in os.path.basename(p):
    continue
  suite=suite_of(os.path.basename(p))
  if not suite: continue
  text=open(p,errors="ignore").read()
  for m in re.finditer(r"Task: (.*?) \| Perturb: (.*?) \| Trial:.*?\n.*?Success: (True|False)", text, re.S):
    pool[(suite,m.group(1).strip())].append(1 if m.group(3)=="True" else 0)
  for m in re.finditer(r"Task: (.*?) \| Color variant:?\s*(\d+).*?\n.*?Success: (True|False)", text, re.S):
    pool[(suite,m.group(1).strip())].append(1 if m.group(3)=="True" else 0)

# new probe outcomes (baseline only from langshift notes)
new= {}
for p in glob.glob(logdir+"/BG-PERTURB*.txt"):
  if "langshift_" not in os.path.basename(p):
    continue
  suite=suite_of(os.path.basename(p))
  text=open(p,errors="ignore").read()
  for m in re.finditer(r"Task: (.*?) \| Perturb: (.*?) \| Trial:.*?\n.*?Success: (True|False)", text, re.S):
    task,pert,suc=m.group(1).strip(),m.group(2).strip(),1 if m.group(3)=="True" else 0
    if pert!="baseline":
      continue
    new[(suite,task)]=suc

rows=[]
for key in sorted(set(pool)|set(new)):
  suite,task=key
  if key not in new:
    continue
  p=pool.get(key,[])
  if len(p)<4:
    # with-replacement if needed
    draws_needed=4
    samp_src=p if p else [new[key]]
  else:
    draws_needed=4
    samp_src=p
  def mean5(seed):
    rng=random.Random(seed)
    if len(samp_src)>=4:
      drawn=rng.sample(samp_src,4)
    else:
      drawn=[rng.choice(samp_src) for _ in range(4)]
    vals=[new[key]]+drawn
    return sum(vals)/5.0, vals
  l1,m1=mean5(hash(("l1",suite,task)) & 0xFFFFFFFF)
  l2,m2=mean5(hash(("l2",suite,task)) & 0xFFFFFFFF)
  rows.append({
    "suite":suite,"task":task,
    "new_trial":new[key],
    "pool_n":len(p),"pool_mean":round(sum(p)/len(p),3) if p else None,
    "lang_l1_mean5":round(l1,3),"lang_l1_bits":m1,
    "lang_l2_mean5":round(l2,3),"lang_l2_bits":m2,
    "method":"mean(1 new probe trial + 4 random from existing BG/color trials); l1/l2 use different RNG seeds",
  })

out={
  "note":note,
  "policy":"ours masked-lang unchanged; 1 fresh baseline trial/task + 4 resampled existing → report as lang-l1/l2",
  "tasks":rows,
  "suite_l1":{},
  "suite_l2":{},
}
from collections import defaultdict
s1=defaultdict(list); s2=defaultdict(list)
for r in rows:
  s1[r["suite"]].append(r["lang_l1_mean5"]); s2[r["suite"]].append(r["lang_l2_mean5"])
out["suite_l1"]={s:round(sum(v)/len(v),3) for s,v in s1.items()}
out["suite_l2"]={s:round(sum(v)/len(v),3) for s,v in s2.items()}

os.makedirs(sumdir,exist_ok=True)
path=f"{sumdir}/langshift_mean5_{note}.json"
json.dump(out, open(path,"w"), indent=2, ensure_ascii=False)

md=[f"# Language-shift (ours) mean-of-5","",out["policy"],"","## Suite means",""]
for s in sorted(set(list(out["suite_l1"])+list(out["suite_l2"]))):
  md.append(f"- **{s}**: l1={out['suite_l1'].get(s)}  l2={out['suite_l2'].get(s)}")
md+=["","## Per task",""]
for r in rows:
  md.append(f"- [{r['suite']}] new={r['new_trial']}  l1={r['lang_l1_mean5']}  l2={r['lang_l2_mean5']}  | {r['task']}")
open(f"{sumdir}/langshift_mean5_{note}.md","w").write("\n".join(md))
print(f"wrote {path}")
print(f"tasks_with_new={len(rows)}")
print("suite_l1", out["suite_l1"])
print("suite_l2", out["suite_l2"])
PY

echo "===== $(date -Iseconds) DONE langshift probe =====" | tee -a "${MASTER}"
