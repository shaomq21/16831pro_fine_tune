"""
RLDS sim-mask preprocessing: apply LIBERO SegmentationRenderEnv masks to RLDS images.

- Replays each episode in sim (BDDL obj_of_interest + instance seg)
- Gripper finger white dots from sim FK
- Optional small mask-edge perturbation on a fraction of frames
- Output: {out_root}/{data_mix}/1.0.0/ (same TFRecord layout as modified_libero_rlds)

Requires conda env with LIBERO + MuJoCo EGL (e.g. subopt):
  export MUJOCO_GL=egl
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_OPENVLA_ROOT))

from libero_sim_mask import LiberoSimMasker  # noqa: E402
from mask_spatial import LIBERO_90_STUDY_SCENE4_TASKS  # noqa: E402
from rlds_mask_state import count_tfrecord_examples  # noqa: E402

DEFAULT_DATA_ROOT = os.environ.get(
    "RLDS_DATA_ROOT",
    "/var/lib/docker/data/checkpoints/fan-test/16831pro_fine_tune/datasets/modified_libero_rlds",
)
DEFAULT_OUT_ROOT = os.environ.get(
    "RLDS_SIM_OUT_ROOT",
    str(_REPO_ROOT / "openvla-oft/datasets/simu_masked_libero_rlds"),
)

DATA_MIX_TO_SUITE = {
    "libero_spatial_no_noops": "libero_spatial",
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_10_no_noops": "libero_10",
    "libero_90_no_noops": "libero_90",
    "libero_90_study_scene4_no_noops": "libero_90",
}

# Filtered subsets: read source RLDS, write under data_mix output name.
FILTERED_MIX = {
    "libero_90_study_scene4_no_noops": {
        "source_mix": "libero_90_no_noops",
        "lang_allowlist": frozenset(LIBERO_90_STUDY_SCENE4_TASKS),
    },
}

TFRECORD_PREFIX = {
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
    "libero_10_no_noops": "libero_10",
    "libero_90_no_noops": "libero_90",
    "libero_90_study_scene4_no_noops": "libero_90_study_scene4",
}

RESUME_FILE = ".rlds_sim_resume.json"
PROGRESS_FILE = ".rlds_sim_mask_progress.json"
SAVE_PROGRESS_EVERY = 1
PROGRESS_WRITE_EVERY_STEPS = 5
NUM_SHARDS = 16


def _ensure_tf_cpu():
    tf.config.set_visible_devices([], "GPU")


def _to_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x


def _decode_str(s):
    if hasattr(s, "decode"):
        return s.decode("utf-8")
    return str(s)


def _append_tfrecord(path: Path, serialized: bytes):
    import struct
    import zlib

    def _masked_crc(data: bytes) -> int:
        return zlib.crc32(data) & 0xFFFFFFFF

    with open(path, "ab") as f:
        f.write(struct.pack("<Q", len(serialized)))
        f.write(struct.pack("<I", _masked_crc(serialized)))
        f.write(serialized)
        f.write(struct.pack("<I", _masked_crc(serialized)))


def _progress_file_path(out_root: Path, data_mix: str, worker_id: int, num_workers: int) -> Path:
    stem = PROGRESS_FILE.replace(".json", "")
    if num_workers > 1:
        return out_root / f"{stem}_{data_mix}_w{worker_id}.json"
    return out_root / f"{stem}_{data_mix}.json"


def _write_progress(path: Path, payload: dict) -> None:
    from datetime import datetime, timezone

    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    tmp.replace(path)


def _episode_lang(episode) -> str:
    steps_ds = episode["steps"]
    if hasattr(steps_ds, "as_numpy_iterator"):
        step0 = next(steps_ds.as_numpy_iterator())
    else:
        step0 = next(iter(steps_ds))
    return _decode_str(step0.get("language_instruction", b"")).lower().strip()


def _count_filtered_episodes(builder, lang_allowlist: frozenset[str]) -> int:
    ds = builder.as_dataset(split="train", shuffle_files=False)
    n = 0
    for ep in ds:
        if _episode_lang(ep) in lang_allowlist:
            n += 1
    return n


def _resolve_mix(args) -> tuple[str, str, frozenset[str] | None]:
    """Return (source_mix, output_mix, lang_allowlist)."""
    output_mix = args.data_mix
    lang_allowlist: frozenset[str] | None = None

    if args.study_scene4:
        output_mix = "libero_90_study_scene4_no_noops"
        lang_allowlist = frozenset(LIBERO_90_STUDY_SCENE4_TASKS)
    elif args.lang_allowlist:
        lang_allowlist = frozenset(x.strip().lower() for x in args.lang_allowlist.split("|") if x.strip())

    if output_mix in FILTERED_MIX:
        spec = FILTERED_MIX[output_mix]
        source_mix = spec["source_mix"]
        lang_allowlist = spec["lang_allowlist"]
    else:
        source_mix = output_mix

    return source_mix, output_mix, lang_allowlist


def _worker_owned_shards(worker_id: int, num_workers: int, n_shards: int):
    if n_shards % num_workers != 0:
        raise ValueError(f"num_shards ({n_shards}) must be divisible by num_workers ({num_workers})")
    return [worker_id + k * num_workers for k in range(n_shards // num_workers)]


def _finalize_multi_worker(out_root: Path, data_mix: str, num_workers: int, n_shards: int, src_dir: Path):
    out_dir = out_root / data_mix / "1.0.0"
    tfrecord_prefix = TFRECORD_PREFIX.get(data_mix, data_mix.replace("_no_noops", ""))
    shard_counts = [0] * n_shards
    for w in range(num_workers):
        counts_path = out_root / f".rlds_sim_shard_counts_{data_mix}_w{w}.json"
        if not counts_path.exists():
            raise FileNotFoundError(f"Missing worker shard counts: {counts_path}")
        with open(counts_path) as f:
            partial = json.load(f)
        for sid, cnt in partial.items():
            shard_counts[int(sid)] += int(cnt)

    for name in ["dataset_info.json", "features.json"]:
        src = src_dir / name
        dst = out_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    info_path = out_dir / "dataset_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        for s in info.get("splits", []):
            if s.get("name") == "train":
                s["shardLengths"] = [str(c) for c in shard_counts]
                break
        with open(info_path, "w") as f:
            json.dump(info, f, indent=1)
    print(f"Finalized {data_mix}: shardLengths={shard_counts}, total={sum(shard_counts)}")


def _mask_episode_steps(
    episode,
    masker: LiberoSimMasker,
    *,
    init_idx: int,
    perturb_prob: float,
    perturb_strength: int,
    rng: np.random.Generator,
    on_step=None,
    ep_state=None,
):
    steps_ds = episode["steps"]
    if hasattr(steps_ds, "as_numpy_iterator"):
        steps_list = list(steps_ds.as_numpy_iterator())
    else:
        steps_list = [s for s in steps_ds]

    lang_raw = steps_list[0].get("language_instruction", b"")
    lang = _decode_str(lang_raw).lower() if lang_raw is not None else ""
    if ep_state is not None:
        ep_state["lang"] = lang[:80]

    actions = np.asarray([_to_numpy(s["action"]) for s in steps_list], dtype=np.float64)
    modified_steps = []
    total_steps = len(steps_list)

    if on_step is not None:
        on_step(0, total_steps)

    masked_frames = masker.iter_masked_rlds_steps(
        lang,
        actions,
        init_idx=init_idx,
        perturb_prob=perturb_prob,
        perturb_strength=perturb_strength,
        rng=rng,
    )

    for step_idx, (step, masked) in enumerate(zip(steps_list, masked_frames)):
        obs_copy = dict(step["observation"])
        obs_copy["image"] = masked
        step_copy = dict(step)
        step_copy["observation"] = obs_copy
        modified_steps.append(step_copy)
        if on_step is not None:
            on_step(step_idx + 1, total_steps)

    return modified_steps


def main():
    try:
        _main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _main():
    parser = argparse.ArgumentParser(description="Apply LIBERO sim masks to RLDS dataset")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--data_mix", type=str, default="libero_spatial_no_noops")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=NUM_SHARDS)
    parser.add_argument("--perturb_prob", type=float, default=0.3, help="Fraction of frames with edge perturbation")
    parser.add_argument("--perturb_strength", type=int, default=2, help="Morphological perturb strength (1-3)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--lang_allowlist",
        type=str,
        default=None,
        help="Pipe-separated language instructions to keep (lowercase)",
    )
    parser.add_argument(
        "--study_scene4",
        action="store_true",
        help="Collect libero_90 STUDY_SCENE4 book tasks only (tasks_info.txt 82-85)",
    )
    args = parser.parse_args()

    source_mix, output_mix, lang_allowlist = _resolve_mix(args)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_dir = out_root / output_mix / "1.0.0"
    src_dir = data_root / source_mix / "1.0.0"

    if args.finalize:
        _finalize_multi_worker(out_root, output_mix, args.num_workers, args.num_shards, src_dir)
        return

    suite = DATA_MIX_TO_SUITE.get(output_mix) or DATA_MIX_TO_SUITE.get(source_mix)
    if suite is None:
        raise ValueError(f"Unknown data_mix {args.data_mix!r}")

    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")

    _ensure_tf_cpu()
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"Loading LiberoSimMasker({suite!r}) source={source_mix} output={output_mix}"
        + (f" filter={len(lang_allowlist)} langs" if lang_allowlist else ""),
        flush=True,
    )
    masker = LiberoSimMasker(suite)
    masker._ensure_libero()
    rng = np.random.default_rng(args.seed + args.worker_id)

    builder = tfds.builder(source_mix, data_dir=str(data_root))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    num_episodes = builder.info.splits["train"].num_examples

    num_workers = args.num_workers
    worker_id = args.worker_id
    if num_workers > 1 and args.num_shards % num_workers != 0:
        raise ValueError(f"--num_shards must be divisible by --num_workers")

    if lang_allowlist:
        if worker_id == 0:
            print(f"Counting filtered episodes ({len(lang_allowlist)} tasks)...", flush=True)
            filtered_total = _count_filtered_episodes(builder, lang_allowlist)
            print(f"Filtered episodes: {filtered_total}", flush=True)
        else:
            filtered_total = None
        num_episodes = filtered_total if filtered_total is not None else num_episodes
    else:
        filtered_total = num_episodes

    if args.max_episodes is not None:
        filtered_total = min(filtered_total if lang_allowlist else num_episodes, args.max_episodes)

    owned_shards = _worker_owned_shards(worker_id, num_workers, args.num_shards) if num_workers > 1 else list(range(args.num_shards))

    if num_workers > 1:
        resume_file = out_root / f".rlds_sim_resume_{output_mix}_w{worker_id}.json"
    else:
        resume_file = out_root / f".rlds_sim_resume_{output_mix}.json"

    resume_from = 0
    done_episodes: set[int] = set()
    if args.resume and resume_file.exists():
        try:
            with open(resume_file) as f:
                state = json.load(f)
            if num_workers > 1:
                done_episodes = set(int(x) for x in state.get("done_episodes", []))
            else:
                resume_from = int(state.get("last_episode", 0))
        except Exception as e:
            print(f"Could not load resume state: {e}")

    if lang_allowlist:
        total_to_process = None  # filled after scan below
    elif num_workers == 1:
        total_to_process = num_episodes - resume_from
        if args.max_episodes is not None:
            total_to_process = min(total_to_process, args.max_episodes)
            ds = ds.take(args.max_episodes)
        if resume_from:
            ds = ds.skip(resume_from)
    else:
        total_to_process = sum(
            1 for ep in range(num_episodes) if ep % num_workers == worker_id and ep not in done_episodes
        )
        if args.max_episodes is not None:
            total_to_process = min(total_to_process, args.max_episodes)

    n_shards = args.num_shards
    tfrecord_prefix = TFRECORD_PREFIX.get(output_mix, output_mix.replace("_no_noops", ""))
    shard_files = [out_dir / f"{tfrecord_prefix}-train.tfrecord-{i:05d}-of-{n_shards:05d}" for i in range(n_shards)]
    writers = {}
    writer_modes = {}
    for sid in owned_shards:
        shard_path = shard_files[sid]
        if args.resume and shard_path.exists() and shard_path.stat().st_size > 0:
            writers[sid] = None
            writer_modes[sid] = "append"
        else:
            writers[sid] = tf.io.TFRecordWriter(str(shard_path))
            writer_modes[sid] = "write"
    shard_counts = {sid: 0 for sid in owned_shards}
    if args.resume:
        for sid in owned_shards:
            shard_counts[sid] = count_tfrecord_examples(shard_files[sid])

    features = builder.info.features
    from tensorflow_datasets.core import example_serializer

    example_serializer_obj = example_serializer.ExampleSerializer(features.get_serialized_info())
    worker_completed = 0
    lane_done = set(done_episodes)
    progress_file = _progress_file_path(out_root, output_mix, worker_id, num_workers)
    lang_init_counter: dict[str, int] = {}

    if lang_allowlist:
        slot = 0
        total_to_process = 0
        scan_ds = builder.as_dataset(split="train", shuffle_files=False)
        for i, ep in enumerate(scan_ds):
            if _episode_lang(ep) not in lang_allowlist:
                continue
            if i in done_episodes:
                continue
            if num_workers == 1 or slot % num_workers == worker_id:
                total_to_process += 1
            slot += 1
        if num_workers == 1:
            total_to_process = max(total_to_process - resume_from, 0)
        if args.max_episodes is not None:
            total_to_process = min(total_to_process, args.max_episodes)

    try:
        desc = f"RLDS sim-mask w{worker_id}" if num_workers > 1 else "RLDS sim-mask"
        tqdm_total = total_to_process if total_to_process is not None else num_episodes
        ep_iter = tqdm(enumerate(ds), total=tqdm_total, desc=desc)
        filtered_slot = 0

        for ep_idx, episode in ep_iter:
            lang = _episode_lang(episode)

            if lang_allowlist:
                if lang not in lang_allowlist:
                    continue
                if ep_idx in lane_done:
                    continue
                my_turn = num_workers == 1 or filtered_slot % num_workers == worker_id
                output_ep = filtered_slot
                filtered_slot += 1
                if not my_turn:
                    continue
                global_ep = ep_idx
            else:
                if num_workers > 1:
                    if ep_idx % num_workers != worker_id:
                        continue
                    global_ep = ep_idx
                    if global_ep in lane_done:
                        continue
                else:
                    global_ep = resume_from + ep_idx
                    if global_ep < resume_from:
                        continue
                output_ep = global_ep

            steps_ds = episode["steps"]
            if hasattr(steps_ds, "as_numpy_iterator"):
                orig_steps = list(steps_ds.as_numpy_iterator())
            else:
                orig_steps = [s for s in steps_ds]
            init_idx = lang_init_counter.get(lang, 0)
            lang_init_counter[lang] = init_idx + 1

            ep_state = {"step": 0, "total": 0, "lang": lang[:80]}

            def on_step(step_idx, step_total):
                ep_state["step"] = step_idx
                ep_state["total"] = step_total
                if step_idx == 0 or step_idx == step_total or step_idx % PROGRESS_WRITE_EVERY_STEPS == 0:
                    _emit_progress(step_idx, phase="masking")

            def _emit_progress(episode_step: int, phase: str = "masking"):
                ep_total = ep_state["total"] or 1
                frac = (worker_completed + (episode_step / ep_total)) / max(total_to_process or 1, 1)
                _write_progress(
                    progress_file,
                    {
                        "data_mix": output_mix,
                        "worker_id": worker_id,
                        "num_workers": num_workers,
                        "total_episodes": total_to_process or num_episodes,
                        "resume_from": resume_from,
                        "total_to_process": total_to_process,
                        "completed_in_run": worker_completed,
                        "global_episode": global_ep,
                        "episode_steps_total": ep_state["total"],
                        "episode_step": episode_step,
                        "episode_progress": episode_step / ep_total if ep_state["total"] else 0.0,
                        "overall_progress": min(frac, 1.0),
                        "language": ep_state["lang"],
                        "phase": phase,
                    },
                )

            modified_steps = _mask_episode_steps(
                episode,
                masker,
                init_idx=init_idx,
                perturb_prob=args.perturb_prob,
                perturb_strength=args.perturb_strength,
                rng=rng,
                on_step=on_step,
                ep_state=ep_state,
            )

            ep_meta_raw = episode.get("episode_metadata")
            ep_metadata = {}
            if ep_meta_raw is not None:
                try:
                    raw = ep_meta_raw.numpy() if hasattr(ep_meta_raw, "numpy") else ep_meta_raw
                    if isinstance(raw, dict):
                        ep_metadata = {k: _to_numpy(v) for k, v in raw.items()}
                    elif isinstance(raw, (bytes, str)):
                        ep_metadata = {"file_path": raw if isinstance(raw, bytes) else raw.encode()}
                except Exception:
                    pass

            modified_episode = {"steps": modified_steps, "episode_metadata": ep_metadata}
            encoded = features.encode_example(modified_episode)
            if isinstance(encoded, bytes):
                serialized = encoded
            elif isinstance(encoded, dict):
                serialized = example_serializer_obj.serialize_example(encoded)
            else:
                serialized = encoded

            shard_id = output_ep % n_shards
            if shard_id not in writer_modes:
                continue
            if writer_modes[shard_id] == "append":
                _append_tfrecord(shard_files[shard_id], serialized)
            else:
                writers[shard_id].write(serialized)
            shard_counts[shard_id] += 1
            worker_completed += 1
            lane_done.add(global_ep)
            _emit_progress(ep_state["total"], phase="writing")

            if worker_completed % SAVE_PROGRESS_EVERY == 0:
                if num_workers > 1:
                    with open(resume_file, "w") as f:
                        json.dump({"completed": len(lane_done), "done_episodes": sorted(lane_done)}, f)
                else:
                    with open(resume_file, "w") as f:
                        json.dump({"last_episode": worker_completed + resume_from}, f)

            if total_to_process is not None and worker_completed >= total_to_process:
                break
    finally:
        for w in writers.values():
            if w is not None:
                w.close()
        masker.close()

    if num_workers == 1 or worker_id == 0:
        for name in ["dataset_info.json", "features.json"]:
            src = src_dir / name
            dst = out_dir / name
            if src.exists():
                shutil.copy2(src, dst)

    if num_workers > 1:
        counts_path = out_root / f".rlds_sim_shard_counts_{output_mix}_w{worker_id}.json"
        with open(counts_path, "w") as f:
            json.dump({str(k): v for k, v in shard_counts.items()}, f)
        with open(resume_file, "w") as f:
            json.dump({"completed": len(lane_done), "done_episodes": sorted(lane_done)}, f)
        print(f"Worker {worker_id} done. Shard counts: {shard_counts}")
    else:
        info_path = out_dir / "dataset_info.json"
        full_counts = [0] * n_shards
        for sid, cnt in shard_counts.items():
            full_counts[sid] = cnt
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            for s in info.get("splits", []):
                if s.get("name") == "train":
                    s["shardLengths"] = [str(c) for c in full_counts]
                    break
            with open(info_path, "w") as f:
                json.dump(info, f, indent=1)
        with open(resume_file, "w") as f:
            json.dump({"last_episode": resume_from + worker_completed}, f)

    print("DONE. Output:", out_dir)


if __name__ == "__main__":
    main()
