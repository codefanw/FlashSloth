# Copyright 2024 Zhenwei Shao and MILVLG team.
# Licensed under the Apache License, Version 2.0.

import logging, os

class VeryUsefulLoggerFormatter(logging.Formatter):
    """ A very useful logger formatter lets you locate where a printed log is coming from.
        This class is written by Zhenwei (https://github.com/ParadoxZW).
    """
    def format(self, record):
        pathname = record.pathname
        parts = pathname.split(os.sep)
        start_idx = max(0, len(parts) - (self.flashsloth_log_fflevel + 1))
        relevant_path = os.sep.join(parts[start_idx:])
        record.custom_path = relevant_path
        return super().format(record)

    @classmethod
    def init_logger_help_function(cls, name, level=logging.INFO):
        flashsloth_silient_others = bool(os.environ.get("FlashSloth_SILIENT_OTHERS", False))
        is_silent = flashsloth_silient_others and os.environ.get("LOCAL_RANK", None) not in ["0", None]
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR if is_silent else level)
        logger.propagate = False
        # customize log format
        log_format = "[%(asctime)s] [%(levelname)s] [%(custom_path)s:%(lineno)d] %(message)s"
        # log_format = "[%(asctime)s] [logger:%(name)s] [%(levelname)s] [%(custom_path)s:%(lineno)d] %(message)s"
        formatter = cls(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        formatter.flashsloth_log_fflevel = int(os.environ.get("flashsloth_LOG_FFLEVEL", "3"))
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger


logger = VeryUsefulLoggerFormatter.init_logger_help_function(__name__)
VeryUsefulLoggerFormatter.init_logger_help_function("", level=logging.WARNING)
VeryUsefulLoggerFormatter.init_logger_help_function("transformers.generation", level=logging.WARNING)
VeryUsefulLoggerFormatter.init_logger_help_function("transformers.modeling_utils", level=logging.ERROR)
# VeryUsefulLoggerFormatter.init_logger_help_function("deepspeed")

try:
    from .model import LlavaLlamaForCausalLM
except:
    pass