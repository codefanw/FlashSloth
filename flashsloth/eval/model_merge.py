import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from flashsloth.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from flashsloth.conversation import conv_templates, SeparatorStyle
from flashsloth.model.builder import load_pretrained_model
from flashsloth.utils import disable_torch_init
from flashsloth.mm_utils import get_model_name_from_path



def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(model_path, args.model_base, model_name)
    model.save_pretrained(f'checkpoints/{args.save_name}/', max_shard_size="1024MB", safe_serialization=True)
    tokenizer.save_pretrained(f'checkpoints/{args.save_name}/')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--save-name", type=str, default='flashsloth-v1-3b')
    args = parser.parse_args()

    eval_model(args)
