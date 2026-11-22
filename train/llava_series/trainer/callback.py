import os

import torch
import torch.distributed as dist
from peft.peft_model import PeftModelForCausalLM
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from llava.train.llava_trainer import maybe_zero_3


def _get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def _get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def _is_rank0(args: TrainingArguments):
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return getattr(args, "local_rank", -1) in (-1, 0)


class LlavaCallback(TrainerCallback):
    """训练结束及中间 step checkpoint 保存时同步落盘完整 LoRA 与 non_lora_trainables.bin / config.json。"""

    def __init__(self, save_steps=50):
        self.save_steps = save_steps

    def _save_to_dir(self, model: PeftModelForCausalLM, target_dir: str, args: TrainingArguments):
        if not _is_rank0(args):
            return
        os.makedirs(target_dir, exist_ok=True)
        print(f"[LlavaCallback] Saving LoRA adapter & non-LoRA state to {target_dir}", flush=True)
        torch.cuda.synchronize()
        safe_serialization = os.environ.get("LLAVA_SAFE_SERIALIZATION", "1") == "1"
        state_dict = _get_peft_state_maybe_zero_3(model.named_parameters(), "none")
        non_lora_state_dict = _get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        model.config.use_cache = True
        model.config.save_pretrained(target_dir)
        model.save_pretrained(target_dir, state_dict=state_dict, safe_serialization=safe_serialization)
        torch.save(non_lora_state_dict, os.path.join(target_dir, "non_lora_trainables.bin"))
        print(f"[LlavaCallback] Finished saving to {target_dir}", flush=True)

    def save_model(self, model: PeftModelForCausalLM, args: TrainingArguments):
        self._save_to_dir(model, args.output_dir, args)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        model = kwargs.get("model")
        if isinstance(model, PeftModelForCausalLM) and state.global_step > 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(ckpt_dir):
                if not _is_rank0(args):
                    return
                model.config.use_cache = True
                model.config.save_pretrained(ckpt_dir)
                non_lora_state_dict = _get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
                torch.save(non_lora_state_dict, os.path.join(ckpt_dir, "non_lora_trainables.bin"))

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        model = kwargs["model"]
        if isinstance(model, PeftModelForCausalLM):
            self.save_model(model, args)

    def on_step_end(self, args, state, control, **kwargs):
        return control
