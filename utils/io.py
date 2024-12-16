import os
import torch
import tempfile
from pathlib import Path
from typing import Any, IO, Union

import logging
logger = logging.Logger(__file__) 


def key_mapping(state_dict, key_mapping_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        flag = 0
        for prev_key in key_mapping_dict.keys():
            if prev_key in k:
                new_state_dict[k.replace(prev_key, key_mapping_dict[prev_key])] = v
                flag = 1
                break
        if flag == 0:
            new_state_dict[k] = v
    return new_state_dict


def partial_load_from_checkpoints(
        local_checkpoint_path, 
        ckpt_rename_parameters=None, 
        map_location="cpu",
        model=None,
        valid_prefix=None,
        lazy_load=False
    ):
    
    ckpt_rename_parameters = ckpt_rename_parameters or dict()
    if os.path.isdir(local_checkpoint_path):
        from safetensors.torch import load
        import multiprocessing
        checkpoint = {}
        files = [file for file in os.listdir(local_checkpoint_path) if file.endswith(".safetensors")]
        if len(files) == 0:
            raise ValueError(f"No safetensors file found in {local_checkpoint_path}")
        file_paths = []
        for file in files:
            file_path = os.path.join(local_checkpoint_path, file)
            if not lazy_load:
                print(f"loading checkpoint from {file_path}")
                with open(file_path, "rb") as f:
                    data = f.read()
                loaded = load(data)
                checkpoint.update(loaded)
            else:
                file_paths.append(file_path)
        if lazy_load:
            return file_paths
    else:
        checkpoint = torch.load(local_checkpoint_path, map_location=map_location)

    if "state_dict" in checkpoint:
        logger.info("partial loading checkpoint")
        state_dict = checkpoint["state_dict"]
    elif "module" in checkpoint:
        # for ds zero2 checkpoint
        logger.info("partial loading deepspeed zero2 checkpoint")
        state_dict = checkpoint["module"]
        ckpt_rename_parameters.update({"module.": ""})
    else:
        state_dict = checkpoint

    if valid_prefix:
        new_state_dict = dict()
        for k, v in state_dict.items():
            for prefix in valid_prefix:
                if k.startswith(prefix):
                    new_state_dict[k] = v
        state_dict = new_state_dict
    state_dict = key_mapping(state_dict, ckpt_rename_parameters)
    return state_dict


def is_lora_checkpoint(state_dict):
    for key in state_dict.keys():
        if "lora" in key:
            return True
    return False
