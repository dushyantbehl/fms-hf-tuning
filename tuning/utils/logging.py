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
from pathlib import Path
from typing import Optional, Self
from accelerate.state import PartialState
import datasets
import transformers
from typing import Dict

DEFAULT_LOG_LEVEL_MAIN = "INFO"
DEFAULT_LOG_LEVEL_WORKERS = "ERROR"

DEFAULT_LOG_FORMAT = (
    "[%(asctime)s %(levelname)-5s]"
    + "[%(filename)20s:%(lineno)3s %(funcName)25s()] %(message)s"
)

__LOGGER_CONFIGURED = False
__LOG_LEVEL = None

def __get_log_level(x: str) -> int:
    """Return the corresponding logging level."""
    d = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
        "passive": logging.WARNING,
    }
    return d[x.lower()]

def get_active_log_level() -> str:
    global __LOG_LEVEL
    return __LOG_LEVEL

class CustomLogFormatter(logging.Formatter):
    """
    Print different log levels in different colors.

    ANSI color codes:
    https://en.wikipedia.org/wiki/ANSI_escape_code
    https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797#ansi-escape-sequences
    https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797#colors--graphics-mode
    https://jvns.ca/blog/2025/03/07/escape-code-standards/#ecma-48
    """

    ESC = "\x1b"
    ESCC = ESC + "["
    DELIM = ";"
    END = "m"

    RESET = "0"
    BOLD = "1"
    DIM = "2"
    BLACK = "30"
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    BLUE = "34"
    MAGENTA = "35"
    CYAN = "36"
    WHITE = "37"

    DO_RESET = ESCC + RESET + END
    DO_DEBUG_COLOR = ESCC + DIM + DELIM + WHITE + END
    DO_INFO_COLOR = ESCC + RESET + DELIM + WHITE + END
    DO_WARNING_COLOR = ESCC + RESET + DELIM + YELLOW + END
    DO_ERROR_COLOR = ESCC + RESET + DELIM + RED + END
    DO_CRITICAL_COLOR = ESCC + BOLD + DELIM + RED + END

    def __init__(self, fmt = None, datefmt = None, style = "%", validate = True, *, defaults = None):
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)

        state = PartialState()
        rank = state.process_index

        format = f"[Rank {rank}]" + DEFAULT_LOG_FORMAT

        self.FORMATS = {
            logging.DEBUG:      self.DO_DEBUG_COLOR     + format + self.DO_RESET,
            logging.INFO:       self.DO_INFO_COLOR      + format + self.DO_RESET,
            logging.WARNING:    self.DO_WARNING_COLOR   + format + self.DO_RESET,
            logging.ERROR:      self.DO_ERROR_COLOR     + format + self.DO_RESET,
            logging.CRITICAL:   self.DO_CRITICAL_COLOR  + format + self.DO_RESET,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter(log_fmt).format(record)

def configure_logging(
    level: str = DEFAULT_LOG_LEVEL_MAIN,
    log_format: Optional[str] = None,
    log_path: Optional[Path] = None,
    skip_if_already_configured: bool = False,
):
    """Configure the basic logger."""
    global __LOGGER_CONFIGURED
    global __LOG_LEVEL

    if skip_if_already_configured and __LOGGER_CONFIGURED:
        return

    #Clear any existing handlers if necessary
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    state = PartialState()
    rank = state.process_index

    if log_path is not None:
        if not isinstance(log_path, Path):
            log_path = Path(log_path)
    elif os.environ.get("LOG_FILE"):
        log_path = Path(os.environ.get("LOG_FILE"))

    if log_path and not log_path.is_absolute():
        log_path = log_path.absolute()
        log_path = log_path.with_name(log_path.stem + f"_rank_{rank}" + log_path.suffix)

    # Configure Python native logger and transformers log level
    # If CLI arg is passed, assign same log level to python native logger
    lowest_level = DEFAULT_LOG_LEVEL_MAIN
    if level is not None:
        lowest_level = level
    elif os.environ.get("LOG_LEVEL"):
        # If CLI arg not is passed and env var LOG_LEVEL is set,
        # assign same log level to both logger
        if os.environ.get("TRANSFORMERS_VERBOSITY"):
            lowest_level = os.environ.get("TRANSFORMERS_VERBOSITY")
        else:
            lowest_level = os.environ.get("LOG_LEVEL")

    if not os.environ.get("LOG_ON_ALL_RANKS"):
        if not state.is_local_main_process:
            lowest_level = logging.DEBUG

    if log_format is not None:
        logging.basicConfig(
            format=log_format,
            level=__get_log_level(lowest_level),
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=log_path,
        )
    else:
        handler: logging.Handler = logging.StreamHandler()
        if log_path is not None:
            handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(CustomLogFormatter())
        logging.basicConfig(
            handlers=[handler],
            level=__get_log_level(lowest_level),
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )

    __LOGGER_CONFIGURED = True
    __LOG_LEVEL = lowest_level
    logger = logging.getLogger(__name__)
    logger.info("logging level is set to %s", level)
    if log_path is not None:
        logger.info("saving logs to file at %s", log_path)

def get_logger(name: str) -> logging.Logger:
    """Set log level of python native logger and TF logger"""
    configure_logging(skip_if_already_configured=True)
    logger = logging.getLogger(name)
    return logger

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
