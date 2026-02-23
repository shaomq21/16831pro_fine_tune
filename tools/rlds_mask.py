"""
RLDS mask preprocessing: apply Grounded-SAM masking to RLDS dataset images.

- Uses the same RLDS pipeline as datasets.py (get_oxe_dataset_kwargs_and_weights, make_interleaved_dataset)
- Resume is based on RLDS iteration index (deterministic order via shuffle=False, shuffle_buffer_size=1)
- Outputs masked images to debug/ for inspection; for 50k+ data you may skip saving all images
"""

from pathlib import Path
import argparse
import json
import os
import sys

from tqdm import tqdm
from PIL import Image

# Resolve paths: run from VLA repo root, openvla-oft is sibling of tools/
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_OPENVLA_ROOT = _REPO_ROOT / "openvla-oft"
if _OPENVLA_ROOT.exists():
    sys.path.insert(0, str(_REPO_ROOT))  # for prismatic
    sys.path.insert(0, str(_OPENVLA_ROOT))  # for mask_processor
else:
    # Fallback for absolute paths (e.g. on server)
    _OPENVLA_ROOT = Path("/home/ubuntu/16831pro_fine_tune/openvla-oft")
    sys.path.insert(0, str(_OPENVLA_ROOT.parent))
    sys.path.insert(0, str(_OPENVLA_ROOT))

from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.oxe import get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.dataset import make_interleaved_dataset

from mask_processor import GroundedSAMConfig, GroundedSAMMasker

# Default paths (override via args or env)
DEFAULT_DATA_ROOT = os.environ.get("RLDS_DATA_ROOT", str(_REPO_ROOT / "openvla-oft/datasets/modified_libero_rlds"))
DEFAULT_OUT_ROOT = os.environ.get("RLDS_OUT_ROOT", str(_REPO_ROOT / "openvla-oft/datasets/masked_libero_rlds"))
RESOLUTION = (224, 224)
RESUME_FILE = ".rlds_resume.json"
SAVE_PROGRESS_EVERY = 100


def _ensure_tf_cpu():
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")


def main():
    parser = argparse.ArgumentParser(description="Apply Grounded-SAM masks to RLDS dataset images")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT, help="RLDS data root")
    parser.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT, help="Output root (debug images saved here)")
    parser.add_argument("--data_mix", type=str, default="libero_goal_no_noops", help="Dataset mix name")
    parser.add_argument("--resume", action="store_true", help="Resume from last processed RLDS index")
    parser.add_argument("--no_save_images", action="store_true", help="Do not save debug images (still process; for 50k+ data)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process (for testing)")
    parser.add_argument("--dino_config", type=str, default=None)
    parser.add_argument("--dino_ckpt", type=str, default=None)
    parser.add_argument("--sam_ckpt", type=str, default=None)
    parser.add_argument("--sam_type", type=str, default="vit_b",
        help="SAM backbone: vit_b (fast) | vit_l | vit_h (slowest, best)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    _ensure_tf_cpu()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_debug = out_root / "debug"
    os.makedirs(out_debug, exist_ok=True)

    # Masker config (vit_b ~10x faster than vit_h, vit_l in between)
    _SAM_CKPT = {
        "vit_b": "sam_vit_b_01ec64.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_h": "sam_vit_h_4b8939.pth",
    }
    dino_config = args.dino_config or str(_OPENVLA_ROOT / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    dino_ckpt = args.dino_ckpt or str(_OPENVLA_ROOT / "groundingdino_swint_ogc.pth")
    sam_type = args.sam_type
    sam_ckpt = args.sam_ckpt or str(_OPENVLA_ROOT / _SAM_CKPT.get(sam_type, "sam_vit_b_01ec64.pth"))
    cfg = GroundedSAMConfig(
        dino_config_path=dino_config,
        dino_checkpoint_path=dino_ckpt,
        sam_checkpoint_path=sam_ckpt,
        sam_type=sam_type,
        device=args.device,
    )

    print("Loading GroundedSAM...")
    masker = GroundedSAMMasker(cfg)

    # RLDS config aligned with datasets.py
    mixture_spec = [(args.data_mix, 1.0)]
    per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
        str(data_root),
        mixture_spec,
        load_camera_views=("primary",),
        load_depth=False,
        load_proprio=False,
        load_language=True,
        action_proprio_normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
    )
    # Deterministic order for resume: no shuffle of files, no shuffle of frames
    for kw in per_dataset_kwargs:
        kw["shuffle"] = False

    rlds_config = dict(
        traj_transform_kwargs=dict(
            window_size=1,
            future_action_window_size=0,
            skip_unlabeled=True,
            goal_relabeling_strategy="uniform",
        ),
        frame_transform_kwargs=dict(
            resize_size=RESOLUTION,
            num_parallel_calls=16,
        ),
        dataset_kwargs_list=per_dataset_kwargs,
        shuffle_buffer_size=1,  # buffer=1 => effectively no shuffle => deterministic
        sample_weights=weights,
        balance_weights=True,
        traj_transform_threads=len(mixture_spec),
        traj_read_threads=len(mixture_spec),
        train=False,  # no augmentation, no repeat; single pass
    )

    dataset, dataset_length, _ = make_interleaved_dataset(**rlds_config)
    print("dataset_length:", dataset_length)

    # Resume state: last processed RLDS index
    resume_file = out_root / RESUME_FILE
    resume_from = 0
    if args.resume and resume_file.exists():
        try:
            with open(resume_file) as f:
                state = json.load(f)
            resume_from = int(state.get("last_index", 0))
            print(f"Resuming from RLDS index {resume_from}")
        except Exception as e:
            print(f"Could not load resume state: {e}")

    # Limit to one epoch and optionally skip already processed
    ds_iterable = dataset.take(dataset_length)
    if resume_from > 0:
        ds_iterable = ds_iterable.skip(resume_from)
    total_to_process = min(dataset_length - resume_from, args.max_samples or (dataset_length - resume_from))
    iterator = ds_iterable.as_numpy_iterator()

    save_images = not args.no_save_images

    for idx, rlds_batch in enumerate(tqdm(iterator, total=total_to_process, desc="RLDS mask")):
        if args.max_samples and idx >= args.max_samples:
            break

        global_idx = resume_from + idx

        # Extract image and language (same structure as datasets.py RLDSBatchTransform)
        img_arr = rlds_batch["observation"]["image_primary"]
        if img_arr.ndim == 4:
            img_arr = img_arr[0]
        img = Image.fromarray(img_arr).convert("RGB")
        lang = rlds_batch["task"]["language_instruction"]
        if hasattr(lang, "decode"):
            lang = lang.decode("utf-8")
        lang = lang.lower()

        out = masker.mask_image_from_lang(img, lang)

        if save_images:
            out_path = out_debug / f"{global_idx:08d}.png"
            out.save(out_path)

        # Save progress for resume
        if (idx + 1) % SAVE_PROGRESS_EVERY == 0:
            with open(resume_file, "w") as f:
                json.dump({"last_index": global_idx + 1}, f)

    # Final progress
    final_idx = resume_from + (args.max_samples if args.max_samples else total_to_process)
    with open(resume_file, "w") as f:
        json.dump({"last_index": final_idx}, f)

    print("DONE. Processed up to RLDS index:", final_idx)
    if save_images:
        print("Debug images saved to:", out_debug)


if __name__ == "__main__":
    main()
