#!/usr/bin/env python3
# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from pathlib import Path
import argparse
import json
import logging
import os
import shlex
import subprocess

# Suppress noisy logging from transitive dependencies (e.g. fm-training-estimator)
# before they are imported.
logging.basicConfig(level=logging.WARNING)

# Suppress loguru DEBUG chatter from the recommender. loguru is used internally
# by tuning_config_recommender; we configure it via its env var before import.
os.environ.setdefault("LOGURU_LEVEL", "INFO")

# Third Party
from tuning_config_recommender.adapters import (  # noqa: E402  # pylint: disable=import-error
    FMSAdapter,
)
import yaml  # noqa: E402

ACCEL_NESTED_PREFIXES = {
    "fsdp_": "fsdp_config",
}

DATA_KEYS = {
    "training_data_path",
    "validation_data_path",
    "dataset",
}


def grab_flags(tokens, start, end):
    cfg, i = {}, start
    while i < end:
        t = tokens[i]
        if t.startswith("--"):
            k, v = t[2:], True
            if "=" in t:
                k, v = k.split("=", 1)
                v = v.strip('"')
            elif i + 1 < end and not tokens[i + 1].startswith("--"):
                v = tokens[i + 1].strip('"')
                i += 1
            cfg[k] = v
        i += 1
    return cfg


def load_yaml(path):
    if path and os.path.exists(path):
        try:
            with open(path, "r") as f:
                y = yaml.safe_load(f)
            return y if isinstance(y, dict) else {}
        except (OSError, yaml.YAMLError):
            return {}
    return {}


def nest_accelerate_flags(flat_dist):
    nested = {section: {} for section in ACCEL_NESTED_PREFIXES.values()}
    remaining = {}

    for k, v in flat_dist.items():
        matched = False
        for prefix, section in ACCEL_NESTED_PREFIXES.items():
            if k.startswith(prefix):
                nested[section][k] = v
                matched = True
                break
        if not matched:
            remaining[k] = v

    for sec in list(nested.keys()):
        if not nested[sec]:
            nested.pop(sec)

    return {**remaining, **nested}


def parse(cmd: str):
    tokens = shlex.split(cmd)
    has_m = "-m" in tokens
    is_accel = "accelerate" in tokens and "launch" in tokens
    if is_accel and has_m:
        m = tokens.index("-m")
        dist_flat = grab_flags(tokens, 0, m)
        train = grab_flags(tokens, m + 2, len(tokens))

    elif has_m:
        m = tokens.index("-m")
        dist_flat = {}
        train = grab_flags(tokens, m + 2, len(tokens))
    else:
        dist_flat = {}
        train = grab_flags(tokens, 0, len(tokens))

    yaml_path = train.pop("data_config", None)
    if yaml_path:
        data = load_yaml(yaml_path)
    else:
        data = {}
    accel_yaml_path = dist_flat.pop("config_file", None)
    accel_yaml = load_yaml(accel_yaml_path) if accel_yaml_path else {}
    dist_nested = nest_accelerate_flags(dist_flat)
    dist = {**accel_yaml, **dist_nested}
    train.pop("config_file", None)

    return train, dist, data


def run_fms_hf(
    command: str | None = None,
    base_dir: str | Path = "tuning_recommender_output/final",
    compute_config: dict | None = None,
    unique_tag: str = "tuning_recommender",
    fsdp_args_format: str | None = None,
    execute: bool = True,
    training_config: dict | None = None,
    distributed_training_config: dict | None = None,
    data_config: dict | None = None,
    tuning_strategy: str = None,
) -> str:
    """Run the FMS tuning config recommender.

    Configs can be supplied either by parsing a *command* string or by passing
    dicts directly via *train_config*, *dist_config*, and *data_config*.
    When both a command string and explicit dicts are provided, the explicit
    dicts are merged on top of (override) the values parsed from the command.

    Args:
        command: Optional training command string to parse
            (e.g. "accelerate launch -m tuning.sft_trainer --model_name_or_path ...").
        base_dir: Directory where config files are written.
        compute_config: Compute resource configuration. When provided,
            fsdp_args_format defaults to "accelerate" instead of "hftrainer".
        unique_tag: Tag for output directory naming.
        fsdp_args_format: FSDP config format. "accelerate" uses accelerate launch
            with config file. "hftrainer" passes FSDP args as HF TrainingArguments
            (for use with torchrun). Defaults to "hftrainer", or "accelerate"
            when compute_config is provided.
        execute: If True (default), execute the generated launch command via
            subprocess.
        training_config: Training arguments dict (e.g. model_name_or_path, lr, …).
        distributed_training_config: Distributed / accelerate arguments dict
            (e.g. num_processes, fsdp_sharding_strategy, …).
        data_config: Data configuration dict
            (e.g. training_data_path, dataset, …).

    Returns:
        The launch command string generated by the recommender.
    """
    # Default fsdp_args_format: "accelerate" when compute_config is provided
    # (multi-node needs accelerate launch), "hftrainer" otherwise.
    if fsdp_args_format is None:
        fsdp_args_format = "accelerate" if compute_config else "hftrainer"
    if command:
        train_cfg, dist_cfg, data_cfg = parse(command)
    else:
        train_cfg, dist_cfg, data_cfg = {}, {}, {}

    # Merge explicit dicts on top of anything parsed from the command string.
    if training_config:
        train_cfg.update(training_config)
    if distributed_training_config:
        dist_cfg.update(distributed_training_config)
    if data_config:
        data_cfg.update(data_config)

    if tuning_strategy:
        training_config["tuning_strategy"] = tuning_strategy

    # fsdp_args_format may arrive inside train_cfg when parsed from the command
    # string (REMAINDER swallows it). Extract it and let it override the kwarg.
    fsdp_args_format = train_cfg.pop("fsdp_args_format", fsdp_args_format)

    # Use output_dir from training config as the base directory for recommender
    # config files, so they live alongside training artifacts.
    if train_cfg.get("output_dir"):
        base_dir = train_cfg["output_dir"]

    train_cfg.pop("config_file", None)
    dist_cfg.pop("config_file", None)

    adapter = FMSAdapter(base_dir=Path(base_dir))
    result = adapter.execute(
        train_cfg,
        dist_cfg,
        compute_config or {},
        data_cfg,
        unique_tag,
        {},
        fsdp_args_format=fsdp_args_format,
    )

    launch_cmd = result["launch_command"]

    if execute:
        subprocess.run(launch_cmd, shell=True, check=True)

    return launch_cmd


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print parsed configs and exit (no adapter, no execution).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run adapter and show launch command but DO NOT execute it.",
    )
    parser.add_argument(
        "--fsdp_args_format",
        choices=["accelerate", "hftrainer"],
        default=None,
        help="FSDP config format: 'accelerate' uses accelerate launch with config file, "
        "'hftrainer' passes FSDP args as HF TrainingArguments (for torchrun). "
        "Defaults to 'hftrainer', or 'accelerate' when compute_config is provided.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        print("Error: No command provided.")
        return

    cmd = " ".join(args.command)

    if args.debug:
        train_cfg, dist_cfg, data_cfg = parse(cmd)
        print("\n[dist_config]\n", json.dumps(dist_cfg, indent=2))
        print("\n[train_config]\n", json.dumps(train_cfg, indent=2))
        print("\n[data_config]\n", json.dumps(data_cfg, indent=2))
        return

    launch_cmd = run_fms_hf(
        command=cmd,
        fsdp_args_format=args.fsdp_args_format,
        execute=not args.preview,
    )

    if args.preview:
        print("\n[LAUNCH COMMAND — PREVIEW ONLY]\n")
        print(launch_cmd)


if __name__ == "__main__":
    main()
