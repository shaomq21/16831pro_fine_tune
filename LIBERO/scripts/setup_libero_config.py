#!/usr/bin/env python3
"""Create ~/.libero/config.yaml with default paths so LIBERO does not prompt (Y/N) on first import."""
import os
import yaml

# Same layout as libero/libero/__init__.py get_default_path_dict()
LIBERO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# libero package is at LIBERO/libero/libero
BENCHMARK_ROOT = os.path.join(LIBERO_ROOT, "libero", "libero")

default_path_dict = {
    "benchmark_root": BENCHMARK_ROOT,
    "bddl_files": os.path.join(BENCHMARK_ROOT, "bddl_files"),
    "init_states": os.path.join(BENCHMARK_ROOT, "init_files"),
    "datasets": os.path.join(BENCHMARK_ROOT, "..", "datasets"),
    "assets": os.path.join(BENCHMARK_ROOT, "assets"),
}

config_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
config_file = os.path.join(config_dir, "config.yaml")

if os.path.exists(config_file):
    print(f"Config already exists: {config_file}")
else:
    os.makedirs(config_dir, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(default_path_dict, f, default_flow_style=False)
    print(f"Created {config_file}")
    for k, v in default_path_dict.items():
        print(f"  {k}: {v}")
