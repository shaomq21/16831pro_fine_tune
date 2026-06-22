"""
RLDS mask preprocessing: apply Grounded-SAM masking to RLDS dataset images.

- Loads raw RLDS episodes via tfds, applies mask to images, writes back to TFRecord format
- Output format: {out_root}/{data_mix}/1.0.0/{tfrecord_prefix}-train.tfrecord-XXXXX-of-YYYYY
- Preserves all fields (actions, state, language, etc.); only images are masked
- Output is directly usable for training (same structure as openvla/modified_libero_rlds)
"""

from pathlib import Path
import argparse
import json
import os
import shutil
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm
from PIL import Image

# Resolve paths: run from VLA repo root, openvla-oft is sibling of tools/
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
sys.path.insert(0, str(_SCRIPT_DIR))
if _OPENVLA_ROOT.exists():
    sys.path.insert(0, str(_REPO_ROOT))  # for prismatic
    sys.path.insert(0, str(_OPENVLA_ROOT))  # for mask_processor
else:
    _OPENVLA_ROOT = Path("/home/ubuntu/16831pro_fine_tune/openvla-oft")
    sys.path.insert(0, str(_OPENVLA_ROOT.parent))
    sys.path.insert(0, str(_OPENVLA_ROOT))

print("[rlds_mask] Importing mask_processor...", flush=True)
from mask_processor import GroundedSAMConfig, GroundedSAMMasker, EpisodeMaskTracker
from sam3_backend import DEFAULT_SAM3_CKPT
from debug_image_prune import DEFAULT_MAX_DEBUG_IMAGES, prune_debug_images
from rlds_mask_state import count_tfrecord_examples
print("[rlds_mask] Imports done.", flush=True)

# Default paths (override via args or env)
DEFAULT_DATA_ROOT = os.environ.get("RLDS_DATA_ROOT", str(_REPO_ROOT / "openvla-oft/datasets/modified_libero_rlds"))
DEFAULT_OUT_ROOT = os.environ.get("RLDS_OUT_ROOT", str(_REPO_ROOT / "openvla-oft/datasets/masked_libero_rlds"))
RESUME_FILE = ".rlds_resume.json"
PROGRESS_FILE = ".rlds_mask_progress.json"
SAVE_PROGRESS_EVERY = 10  # episodes
NUM_SHARDS = 16
# Map data_mix name to tfrecord filename prefix (from dataset_info.json "name")
TFRECORD_PREFIX = {
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
    "libero_90_no_noops": "libero_90",
    "libero_10_no_noops": "libero_10",
}


def _ensure_tf_cpu():
    tf.config.set_visible_devices([], "GPU")


def _to_numpy(x):
    """Convert tf.Tensor to numpy if needed."""
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x


def _append_tfrecord(path: Path, serialized: bytes):
    """Append one TFRecord example to an existing shard file."""
    import struct
    import zlib

    def _masked_crc(data: bytes) -> int:
        return zlib.crc32(data) & 0xFFFFFFFF

    with open(path, "ab") as f:
        f.write(struct.pack("<Q", len(serialized)))
        f.write(struct.pack("<I", _masked_crc(serialized)))
        f.write(serialized)
        f.write(struct.pack("<I", _masked_crc(serialized)))


def _decode_str(s):
    if hasattr(s, "decode"):
        return s.decode("utf-8")
    return str(s)


def _safe_lang_name(lang: str) -> str:
    return lang.strip().lower().replace(" ", "_").replace("/", "_")[:80]


def _debug_frame_indices(n_steps: int, n_frames: int) -> list[int]:
    if n_steps <= 0:
        return []
    if n_frames <= 1:
        return [0]
    return sorted(
        {min(n_steps - 1, max(0, int(round(i * (n_steps - 1) / (n_frames - 1))))) for i in range(n_frames)}
    )


def _save_episode_debug(
    debug_dir: Path,
    worker_id: int,
    global_ep: int,
    lang: str,
    orig_steps: list,
    modified_steps: list,
    n_frames: int,
    max_images: int = DEFAULT_MAX_DEBUG_IMAGES,
) -> None:
    """Save raw + masked PNGs for a few frames of one episode."""
    n = min(len(orig_steps), len(modified_steps))
    if n <= 0:
        return
    task_dir = debug_dir / f"w{worker_id}" / _safe_lang_name(lang)
    task_dir.mkdir(parents=True, exist_ok=True)
    for fi in _debug_frame_indices(n, n_frames):
        raw_arr = _to_numpy(orig_steps[fi]["observation"].get("image"))
        masked_arr = modified_steps[fi]["observation"].get("image")
        if raw_arr is None or masked_arr is None or raw_arr.size == 0:
            continue
        Image.fromarray(raw_arr).convert("RGB").save(task_dir / f"ep{global_ep:05d}_frame{fi:03d}_raw.png")
        Image.fromarray(masked_arr).convert("RGB").save(task_dir / f"ep{global_ep:05d}_frame{fi:03d}_masked.png")
    prune_debug_images(debug_dir, max_images)


def _progress_file_path(out_root: Path, data_mix: str, worker_id: int, num_workers: int) -> Path:
    if num_workers > 1:
        return out_root / f"{PROGRESS_FILE.replace('.json', '')}_{data_mix}_w{worker_id}.json"
    return out_root / f"{PROGRESS_FILE.replace('.json', '')}_{data_mix}.json"


def _write_progress(path: Path, payload: dict) -> None:
    from datetime import datetime, timezone

    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    tmp.replace(path)


def _mask_episode_steps(episode, masker, mask_wrist=True, use_temporal=True, on_step=None, ep_state=None):
    """Apply mask to image and wrist_image in each step. Returns modified steps list."""
    steps_ds = episode["steps"]
    if hasattr(steps_ds, "as_numpy_iterator"):
        steps_list = list(steps_ds.as_numpy_iterator())
    else:
        steps_list = [s for s in steps_ds]

    lang_raw = steps_list[0].get("language_instruction", b"")
    lang = _decode_str(lang_raw).lower() if lang_raw is not None else ""
    if ep_state is not None:
        ep_state["lang"] = lang[:80]

    tracker = EpisodeMaskTracker(masker) if use_temporal else None
    wrist_tracker = EpisodeMaskTracker(masker) if use_temporal else None

    modified_steps = []
    total_steps = len(steps_list)
    if on_step is not None:
        on_step(0, total_steps)
    for step_idx, step in enumerate(steps_list):
        obs = dict(step["observation"])
        proprio = _to_numpy(obs.get("state"))
        joint = _to_numpy(obs.get("joint_state"))
        # Mask primary image (temporal SAM3 tracking)
        img_arr = _to_numpy(obs["image"])
        if img_arr is not None and img_arr.size > 0:
            img = Image.fromarray(img_arr).convert("RGB")
            if tracker is not None:
                masked = tracker.mask_image_from_lang(
                    img, lang, proprio_state=proprio, joint_state=joint
                )
            else:
                masked = masker.mask_image_from_lang(
                    img, lang, proprio_state=proprio, joint_state=joint
                )
            obs["image"] = np.array(masked)
        # Mask wrist image (separate temporal state)
        if mask_wrist and "wrist_image" in obs:
            wrist_arr = _to_numpy(obs["wrist_image"])
            if wrist_arr is not None and wrist_arr.size > 0:
                wrist_img = Image.fromarray(wrist_arr).convert("RGB")
                if wrist_tracker is not None:
                    masked_wrist = wrist_tracker.mask_image_from_lang(wrist_img, lang)
                else:
                    masked_wrist = masker.mask_image_from_lang(wrist_img, lang)
                obs["wrist_image"] = np.array(masked_wrist)
        step_copy = dict(step)
        step_copy["observation"] = obs
        modified_steps.append(step_copy)
        if on_step is not None:
            on_step(step_idx + 1, total_steps)
    return modified_steps


def main():
    import traceback
    try:
        _main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


def _worker_owned_shards(worker_id: int, num_workers: int, n_shards: int):
    """Shard IDs this worker writes to (ep % num_workers == worker_id => ep % n_shards)."""
    if n_shards % num_workers != 0:
        raise ValueError(f"num_shards ({n_shards}) must be divisible by num_workers ({num_workers})")
    return [worker_id + k * num_workers for k in range(n_shards // num_workers)]


def _finalize_multi_worker(out_root: Path, data_mix: str, num_workers: int, n_shards: int, src_dir: Path):
    """Merge per-worker shard counts into dataset_info.json after all workers finish."""
    out_dir = out_root / data_mix / "1.0.0"
    tfrecord_prefix = TFRECORD_PREFIX.get(data_mix, data_mix.replace("_no_noops", ""))
    shard_counts = [0] * n_shards
    for w in range(num_workers):
        counts_path = out_root / f".rlds_shard_counts_{data_mix}_w{w}.json"
        if not counts_path.exists():
            raise FileNotFoundError(f"Missing worker shard counts: {counts_path} (worker {w} not done?)")
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
    print(f"Finalized {data_mix}: shardLengths={shard_counts}")
    print(f"Total episodes: {sum(shard_counts)}")


def _main():
    print("[rlds_mask] _main() started.", flush=True)
    parser = argparse.ArgumentParser(description="Apply Grounded-SAM masks to RLDS dataset; output TFRecord for training")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="RLDS data root (input)")
    parser.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT, help="Output root for masked TFRecord")
    parser.add_argument("--data_mix", type=str, default="libero_goal_no_noops", help="Dataset name (e.g. libero_goal_no_noops)")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed episode index")
    parser.add_argument("--no_mask_wrist", action="store_true", help="Skip masking wrist camera (only mask primary)")
    parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to process (for testing)")
    parser.add_argument("--num_shards", type=int, default=NUM_SHARDS, help="Number of output TFRecord shards")
    parser.add_argument("--dino_config", type=str, default=None)
    parser.add_argument("--dino_ckpt", type=str, default=None)
    parser.add_argument("--sam_ckpt", type=str, default=None)
    parser.add_argument("--sam_type", type=str, default="vit_b",
        help="SAM1 backbone: vit_b | vit_l | vit_h (only if --sam_backend sam1)")
    parser.add_argument("--sam_backend", type=str, default=None, choices=["sam1", "sam3"],
        help="Segmentation backend: sam1 (libero_goal default) or sam3 (spatial default)")
    parser.add_argument("--sam3_ckpt", type=str, default=None, help="Path to sam3.pt weights")
    parser.add_argument("--no_temporal", action="store_true", help="Disable per-episode temporal SAM3 tracking")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip SAM3 video fallback, fewer retries, no disk saves")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug_dir", type=str, default=None, help="Save raw+masked debug PNGs (default: rlds_mask_debug/<data_mix>)")
    parser.add_argument("--debug_every_episodes", type=int, default=1, help="Save debug frames every N completed episodes (default 1)")
    parser.add_argument("--debug_frames", type=int, default=1, help="Frames per debug episode (default 1)")
    parser.add_argument("--max_debug_images", type=int, default=DEFAULT_MAX_DEBUG_IMAGES, help="Max debug PNGs kept (default 20)")
    parser.add_argument("--debug_every", type=int, default=None, help=argparse.SUPPRESS)  # legacy alias
    parser.add_argument("--worker_id", type=int, default=0, help="Worker index for multi-GPU parallel (0 .. num_workers-1)")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel GPU workers (must divide num_shards)")
    parser.add_argument("--finalize", action="store_true", help="Merge per-worker outputs into dataset_info.json (run after all workers finish)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_dir = out_root / args.data_mix / "1.0.0"
    src_dir = data_root / args.data_mix / "1.0.0"

    if args.finalize:
        _finalize_multi_worker(out_root, args.data_mix, args.num_workers, args.num_shards, src_dir)
        return

    _ensure_tf_cpu()

    tfrecord_prefix = TFRECORD_PREFIX.get(args.data_mix, args.data_mix.replace("_no_noops", ""))
    if not src_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dir}")

    os.makedirs(out_dir, exist_ok=True)
    if args.debug_dir is None:
        args.debug_dir = f"rlds_mask_debug/{args.data_mix}"
    os.makedirs(args.debug_dir, exist_ok=True)

    # Masker
    _SAM_CKPT = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }
    dino_config = args.dino_config or str(_OPENVLA_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    dino_ckpt = args.dino_ckpt or str(_OPENVLA_ROOT / "groundingdino_swint_ogc.pth")
    sam_ckpt = args.sam_ckpt or str(_OPENVLA_ROOT / _SAM_CKPT.get(args.sam_type, "sam_vit_b_01ec64.pth"))
    sam3_ckpt = args.sam3_ckpt or DEFAULT_SAM3_CKPT
    sam_backend = args.sam_backend or (
        "sam1" if "libero_goal" in args.data_mix else "sam3"
    )
    cfg = GroundedSAMConfig(
        dino_config_path=dino_config,
        dino_checkpoint_path=dino_ckpt,
        sam_checkpoint_path=sam_ckpt,
        sam_type=args.sam_type,
        sam_backend=sam_backend,
        sam3_checkpoint_path=sam3_ckpt,
        device=args.device,
        fast_mode=args.fast,
    )
    print(f"Loading masker (backend={sam_backend}, worker={args.worker_id}/{args.num_workers}, fast={args.fast})...", flush=True)
    masker = GroundedSAMMasker(cfg)
    use_temporal = (sam_backend == "sam3") and (not args.no_temporal)
    print(f"GroundedSAM loaded. temporal={use_temporal}", flush=True)

    # Load RLDS at episode level
    builder = tfds.builder(args.data_mix, data_dir=str(data_root))
    ds = builder.as_dataset(split="train", shuffle_files=False)
    num_episodes = builder.info.splits["train"].num_examples
    if args.max_episodes is not None:
        num_episodes = min(num_episodes, args.max_episodes)
    print("Total episodes:", num_episodes)

    num_workers = args.num_workers
    worker_id = args.worker_id
    if num_workers > 1 and args.num_shards % num_workers != 0:
        raise ValueError(f"--num_shards ({args.num_shards}) must be divisible by --num_workers ({num_workers})")

    owned_shards = (
        _worker_owned_shards(worker_id, num_workers, args.num_shards)
        if num_workers > 1
        else list(range(args.num_shards))
    )
    episodes_for_worker = sum(
        1 for ep in range(num_episodes) if (num_workers == 1 or ep % num_workers == worker_id)
    )
    if args.max_episodes is not None:
        episodes_for_worker = min(episodes_for_worker, args.max_episodes)
    print(f"Worker {worker_id} owns shards {owned_shards}, will process {episodes_for_worker} episodes")

    # Resume
    if num_workers > 1:
        resume_file = out_root / f".rlds_resume_{args.data_mix}_w{worker_id}.json"
    else:
        resume_file = out_root / f".rlds_resume_{args.data_mix}.json"
    resume_skip = 0
    resume_from = 0
    done_episodes: set[int] = set()
    if args.resume and resume_file.exists():
        try:
            with open(resume_file) as f:
                state = json.load(f)
            if num_workers > 1:
                done_episodes = set(int(x) for x in state.get("done_episodes", []))
                resume_skip = int(state.get("completed", 0))
                print(
                    f"Resuming worker {worker_id}: {len(done_episodes)} episodes already in TFRecord"
                    if done_episodes
                    else f"Resuming worker {worker_id}: skip {resume_skip} episodes"
                )
            else:
                resume_from = int(state.get("last_episode", 0))
                ds = ds.skip(resume_from)
                resume_skip = 0  # handled via skip above
                print(f"Resuming from episode {resume_from}")
        except Exception as e:
            print(f"Could not load resume state: {e}")
            resume_from = 0
    else:
        resume_from = 0

    if num_workers == 1:
        total_to_process = num_episodes - resume_from
        if args.max_episodes is not None:
            total_to_process = min(total_to_process, args.max_episodes)
            ds = ds.take(args.max_episodes)
    else:
        total_to_process = sum(
            1
            for ep in range(num_episodes)
            if ep % num_workers == worker_id and ep not in done_episodes
        )
        if args.max_episodes is not None:
            total_to_process = min(total_to_process, args.max_episodes)

    # Open shard writers (multi-GPU: only owned shards to avoid write conflicts)
    n_shards = args.num_shards
    shard_files = [
        out_dir / f"{tfrecord_prefix}-train.tfrecord-{i:05d}-of-{n_shards:05d}"
        for i in range(n_shards)
    ]
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
    debug_every_episodes = args.debug_every_episodes
    if args.debug_every is not None:
        debug_every_episodes = max(1, args.debug_every // 100)
    debug_dir = Path(args.debug_dir)
    worker_completed = 0
    lane_done = set(done_episodes)
    progress_file = _progress_file_path(out_root, args.data_mix, worker_id, num_workers)

    try:
        desc = f"RLDS mask w{worker_id}" if num_workers > 1 else "RLDS mask"
        ep_iter = enumerate(ds)
        if num_workers == 1:
            ep_iter = tqdm(ep_iter, total=total_to_process, desc=desc)
        else:
            ep_iter = tqdm(ep_iter, total=num_episodes, desc=desc)

        for ep_idx, episode in ep_iter:
            if num_workers > 1:
                if ep_idx % num_workers != worker_id:
                    continue
                global_ep = ep_idx
                if global_ep in lane_done:
                    if not done_episodes and worker_completed < resume_skip:
                        worker_completed += 1
                    continue
            else:
                global_ep = resume_from + ep_idx

            ep_state = {"step": 0, "total": 0, "lang": ""}
            steps_ds = episode["steps"]
            if hasattr(steps_ds, "as_numpy_iterator"):
                orig_steps = list(steps_ds.as_numpy_iterator())
            else:
                orig_steps = [s for s in steps_ds]

            def on_step(step_idx, step_total):
                ep_state["step"] = step_idx
                ep_state["total"] = step_total
                _emit_progress(step_idx, phase="masking")

            def _emit_progress(episode_step: int, phase: str = "masking"):
                ep_total = ep_state["total"] or 1
                frac = (worker_completed + (episode_step / ep_total)) / max(total_to_process, 1)
                _write_progress(
                    progress_file,
                    {
                        "data_mix": args.data_mix,
                        "worker_id": worker_id,
                        "num_workers": num_workers,
                        "total_episodes": num_episodes,
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
                episode, masker, mask_wrist=not args.no_mask_wrist, use_temporal=use_temporal,
                on_step=on_step,
                ep_state=ep_state,
            )
            steps_this_ep = len(modified_steps)
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
            modified_episode = {
                "steps": modified_steps,
                "episode_metadata": ep_metadata,
            }
            encoded = features.encode_example(modified_episode)
            if isinstance(encoded, bytes):
                serialized = encoded
            elif isinstance(encoded, dict):
                serialized = example_serializer_obj.serialize_example(encoded)
            else:
                serialized = encoded
            shard_id = global_ep % n_shards
            if writer_modes[shard_id] == "append":
                _append_tfrecord(shard_files[shard_id], serialized)
            else:
                writers[shard_id].write(serialized)
            shard_counts[shard_id] += 1
            worker_completed += 1
            lane_done.add(global_ep)
            _emit_progress(ep_state["total"], phase="writing")

            if debug_dir and (worker_completed == 1 or worker_completed % debug_every_episodes == 0):
                lang_dbg = ep_state["lang"] or _decode_str(
                    orig_steps[0].get("language_instruction", b"")
                ).lower()
                _save_episode_debug(
                    debug_dir,
                    worker_id,
                    global_ep,
                    lang_dbg,
                    orig_steps,
                    modified_steps,
                    args.debug_frames,
                    args.max_debug_images,
                )

            if worker_completed % SAVE_PROGRESS_EVERY == 0:
                if num_workers > 1:
                    with open(resume_file, "w") as f:
                        json.dump(
                            {
                                "completed": len(lane_done),
                                "done_episodes": sorted(lane_done),
                            },
                            f,
                        )
                else:
                    with open(resume_file, "w") as f:
                        json.dump({"last_episode": global_ep + 1}, f)

            if num_workers > 1 and worker_completed >= total_to_process:
                break
    finally:
        for sid, w in writers.items():
            if w is not None:
                w.close()

    # Copy metadata (worker 0 only in multi-GPU mode)
    if num_workers == 1 or worker_id == 0:
        for name in ["dataset_info.json", "features.json"]:
            src = src_dir / name
            dst = out_dir / name
            if src.exists():
                shutil.copy2(src, dst)
                print(f"Copied {name} to {dst}")
            else:
                print(f"Warning: {src} not found, skipping")

    if num_workers > 1:
        counts_path = out_root / f".rlds_shard_counts_{args.data_mix}_w{worker_id}.json"
        with open(counts_path, "w") as f:
            json.dump({str(k): v for k, v in shard_counts.items()}, f)
        with open(resume_file, "w") as f:
            json.dump(
                {"completed": len(lane_done), "done_episodes": sorted(lane_done)},
                f,
            )
        print(f"Worker {worker_id} done. Shard counts: {shard_counts}")
        print(f"Run --finalize after all {num_workers} workers finish.")
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
            json.dump({"last_episode": resume_from + total_to_process}, f)
        _write_progress(
            progress_file,
            {
                "data_mix": args.data_mix,
                "worker_id": worker_id,
                "num_workers": num_workers,
                "total_episodes": num_episodes,
                "resume_from": resume_from,
                "total_to_process": total_to_process,
                "completed_in_run": total_to_process,
                "global_episode": resume_from + total_to_process,
                "episode_steps_total": 0,
                "episode_step": 0,
                "episode_progress": 1.0,
                "overall_progress": 1.0,
                "language": "",
                "phase": "done",
            },
        )

    print("DONE. Output:", out_dir)
    if num_workers == 1:
        print("TFRecord files:", [str(p) for p in shard_files])
    print("Directly usable for training with --data_root", out_root)


if __name__ == "__main__":
    main()
