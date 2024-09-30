import os
import json
import yaml
from typing import Union, Sequence, Mapping

import torch
import rclone

import logging

logger = logging.getLogger(__name__)

def get_extension(file_path: str) -> str:
    _, ext = os.path.splitext(file_path)
    return ext.lower()

def load_yaml_or_json(file_path: str) -> dict:
    with open(file_path, "r") as f:
        match get_extension(file_path):
            case ".yaml" | ".yml":
                return yaml.safe_load(f)
            case ".json":
                return json.load(f)
    return None
