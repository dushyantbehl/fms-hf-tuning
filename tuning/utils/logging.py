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
import logging
import os
from accelerate.state import PartialState
import datasets
import transformers
from typing import Dict

DEFAULT_LOG_LEVEL_MAIN = "INFO"
DEFAULT_LOG_LEVEL_WORKERS = "ERROR"

TRAIN_LOGGER = None
TRAIN_LOGGER_LEVEL = None

def get_logger(logger_name="fms-hf-tuning", train_args_loglevel=None):
    """Set log level of python native logger and TF logger via argument from CLI or env variable.

    Args:
        train_args
            Training arguments for training model.
        logger_name
            Logger name with which the logger is instantiated.

    Returns:
        train_args
            Updated training arguments for training model.
        train_logger
            Logger with updated effective log level
    """

    global TRAIN_LOGGER
    global TRAIN_LOGGER_LEVEL
    if TRAIN_LOGGER and TRAIN_LOGGER_LEVEL:
        return TRAIN_LOGGER, TRAIN_LOGGER_LEVEL

    # Clear any existing handlers if necessary
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure Python native logger and transformers log level
    # If CLI arg is passed, assign same log level to python native logger
    lowest_log_level = DEFAULT_LOG_LEVEL_MAIN
    if train_args_loglevel != "passive":
        lowest_log_level = train_args_loglevel
    elif os.environ.get("LOG_LEVEL"):
        # If CLI arg not is passed and env var LOG_LEVEL is set,
        # assign same log level to both logger
        lowest_log_level = (
            os.environ.get("LOG_LEVEL").lower()
            if not os.environ.get("TRANSFORMERS_VERBOSITY")
            else os.environ.get("TRANSFORMERS_VERBOSITY")
        )

    state = PartialState()
    rank = state.process_index

    log_on_all = os.environ.get("LOG_ON_ALL_PROCESSES")
    if log_on_all:
        log_level = lowest_log_level
    else:
        if state.is_local_main_process:
            log_level = lowest_log_level
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            log_level = DEFAULT_LOG_LEVEL_WORKERS
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

    log_format = f"Rank-{rank} [%(levelname)s]:%(filename)s:%(funcName)s: %(message)s"

    logging.basicConfig(
        format=log_format, level=log_level.upper(),
    )

    if logger_name:
        train_logger = logging.getLogger(logger_name)
    else:
        train_logger = logging.getLogger()

    TRAIN_LOGGER = train_logger
    TRAIN_LOGGER_LEVEL = log_level.lower()

    return TRAIN_LOGGER, TRAIN_LOGGER_LEVEL

def pretty_print_args(args: Dict):
    dump = "\n========================= Flat Arguments =========================\n"
    for name, arg in args.items():
        if arg:
            dump += f"---------------------------- {name} -----------------------\n"
            if hasattr(arg, '__dict__'):
                arg = vars(arg)
            max_len = max(len(k) for k in arg.keys())
            for k, v in sorted(arg.items()):
                dump += f"  {k:<{max_len}} : {v}\n"
    dump += "========================= Arguments Done =========================\n"
    return dump
