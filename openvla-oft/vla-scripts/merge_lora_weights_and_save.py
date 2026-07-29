"""
Merge a vision_backbone LoRA adapter into the base OpenVLA and save the full model.

Adapters in this repo are saved from `vla.vision_backbone` only (see finetune.py).
We therefore load/merge on `vision_backbone`, then save the full VLA checkpoint.

Usage:
    python vla-scripts/merge_lora_weights_and_save.py \
        --base_checkpoint /path/to/openvla-7b \
        --lora_finetuned_checkpoint_dir /path/to/run_dir/
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class ConvertConfig:
    # fmt: off
    base_checkpoint: Union[str, Path] = ""
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""
    adapter_subdir: str = "lora_adapter"
    # fmt: on


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    ckpt_dir = Path(cfg.lora_finetuned_checkpoint_dir)
    adapter_dir = ckpt_dir / cfg.adapter_subdir
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter at {adapter_dir}")

    print(f"Loading base model: {cfg.base_checkpoint}")
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.base_checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"Merging LoRA weights from {adapter_dir} into vision_backbone...")
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vla.vision_backbone = PeftModel.from_pretrained(vla.vision_backbone, str(adapter_dir))
    vla.vision_backbone = vla.vision_backbone.merge_and_unload()
    vla = vla.to(device)

    print(f"Saving merged model to {ckpt_dir} ...")
    vla.save_pretrained(ckpt_dir)
    print(f"\nMerging complete! Time elapsed (sec): {time.time() - start_time}")
    print(f"Saved merged model checkpoint at:\n{ckpt_dir}")


if __name__ == "__main__":
    main()
