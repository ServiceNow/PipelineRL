import json
import logging
import os
import sys
from pathlib import Path

import torch
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraConfig
from peft.utils.other import prepare_model_for_kbit_training
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def has_lora_checkpoint(current_dir: Path) -> bool:
    return os.path.exists(current_dir / "adapter_model.bin") or os.path.exists(
        current_dir / "adapter_model.safetensors"
    )


def prepare_model_for_bf16_training(model, use_gradient_checkpointing=False, gradient_checkpointing_kwargs=None):
    logger.info("Prepare LoRA for BF16 training")
    for _, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    for name, param in model.named_parameters():
        # upcast LM head and layernorms
        if any([k in name for k in ["lm_head", "wte", "ln_"]]):
            param.data = param.data.to(torch.float32)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()

    return model


def prepare_lora_model(lora_config, model, gradient_checkpointing) -> PeftModel:
    logger.info("Prepare LoRA adapters")
    lora_prepare_fn = (
        prepare_model_for_kbit_training
        if lora_config.base_model_8bit or lora_config.base_model_4bit
        else prepare_model_for_bf16_training
    )
    all_params = model.num_parameters()
    model = lora_prepare_fn(
        model, use_gradient_checkpointing=gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    lora_config = LoraConfig(
        inference_mode=False,
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        target_modules=list(lora_config.target_modules),
    )
    model = get_peft_model(model, lora_config)
    trainable_params = model.num_parameters(only_trainable=True)
    logger.info(
        f"LoRA trainable params: {trainable_params:_} of {all_params:_} || trainable%: {100 * trainable_params / all_params:.2f}"
    )
    return model  # type: ignore


def is_lora_checkpoint(lora_model_path):
    lora_model_config = os.path.join(lora_model_path, "adapter_config.json")
    return os.path.exists(lora_model_config)


def get_base_model_name(lora_model_path):
    assert is_lora_checkpoint(lora_model_path)
    lora_model_config = os.path.join(lora_model_path, "adapter_config.json")
    with open(lora_model_config) as f:
        lora_config = json.load(f)
    return lora_config["base_model_name_or_path"]


def lora_load_and_merge(lora_model_path, **kwargs):
    base_model_name_or_path = get_base_model_name(lora_model_path)
    logger.info(f"Load base model {base_model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, **kwargs)
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    logger.info(f"Merge adapters from {lora_model_path}")
    return model.merge_and_unload()  # type: ignore


def lora_save(checkpoint_folder, model):
    model.save_pretrained(checkpoint_folder)
    logger.info(f"Saved LoRA model to {checkpoint_folder}")


def lora_load(checkpoint_folder, model):
    adapter_path = os.path.join(checkpoint_folder, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        adapters_weights = load_file(adapter_path)
    else:
        adapter_path = os.path.join(checkpoint_folder, "adapter_model.bin")
        adapters_weights = torch.load(adapter_path)
    set_peft_model_state_dict(model, adapters_weights)
    logger.info(f"Loaded LoRA model from {checkpoint_folder}.")
    return model


def apply_lora(model, lora_model_path):
    lora_model_config = os.path.join(lora_model_path, "adapter_config.json")
    assert os.path.exists(lora_model_config)
    with open(lora_model_config) as f:
        lora_config = json.load(f)
    scaling = lora_config["lora_alpha"] / lora_config["r"]
    lora_model_file = os.path.join(lora_model_path, "adapter_model.bin")
    assert os.path.exists(lora_model_file)
    lora_weights = torch.load(lora_model_file)
    for layer_name, layer_weights in model.named_parameters():
        lora_a = None
        lora_b = None
        for k, v in lora_weights.items():
            if layer_name in k:
                if "lora_A" in k:
                    lora_a = v
                elif "lora_B" in k:
                    lora_b = v
            if lora_a is not None and lora_b is not None:
                wdiff = (lora_b @ lora_a) * scaling
                layer_weights.data += wdiff
                break


def merge_lora(lora_model_path):
    if lora_model_path[-1] == "/":
        lora_model_path = lora_model_path[:-1]
    assert os.path.isdir(lora_model_path), f"{lora_model_path} is not a dir"
    lora_model_config = os.path.join(lora_model_path, "adapter_config.json")
    assert os.path.exists(lora_model_config), f"{lora_model_config} does not exists"

    logger.info(f"Merge lora checkpoint {lora_model_path}")
    model = lora_load_and_merge(lora_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(lora_model_path)

    tmp_dir = f"{lora_model_path}_merged"
    logger.info(f"Save merged model to {tmp_dir}")
    model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)

    os.rename(lora_model_path, f"{lora_model_path}_lora")
    os.rename(tmp_dir, lora_model_path)
    logger.info(f"Merged model saved to {lora_model_path}")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Merging lora weights: python lora.py <lora_weights_dir>"
    merge_lora(sys.argv[1])
