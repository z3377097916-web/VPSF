# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import Any, Callable, Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from datasets import Dataset
else:
    Dataset = Any
from packaging import version
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl import DPOConfig, create_reference_model

from utils import (
    DPODataCollatorWithPaddingAndImage,
    PreTrainedModelWrapper,
    disable_dropout_in_model,
)


class BaseDPOTrainer(Trainer):
    "Base LVLM DPO Trainer."

    def _load_rng_state(self, checkpoint):


        import numpy as np
        with torch.serialization.safe_globals([
            np.core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype(np.uint32)),
        ]):
            return super()._load_rng_state(checkpoint)

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: DPOConfig = None,
        data_collator=None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        is_encoder_decoder: bool = False,
        dpo_beta: Optional[float] = 0.1,
        gamma: Optional[float] = 0.0,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        disable_dropout: bool = True,
        compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
    ):
        self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = (
            isinstance(model, PeftModel)
            or (hasattr(model, "llama_model") and isinstance(model.llm_model, PeftModel))
            or (hasattr(model, "llm_model") and isinstance(model.llm_model, PeftModel))
        )

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:

            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)  # noqa
        if data_collator is None:
            if processing_class is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            if max_target_length is None and self.is_encoder_decoder:
                warnings.warn(
                    "When using DPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_target_length = 128

            data_collator = DPODataCollatorWithPaddingAndImage()

            if args.remove_unused_columns:
                args.remove_unused_columns = False

                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.ref_adapter_name = args.ref_adapter_name

        self.gamma = gamma
        self.dpo_beta = dpo_beta

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.model_accepts_loss_kwargs = False

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if self.ref_model is None:
            if hasattr(model, "llama_model"):
                if not hasattr(self.accelerator.unwrap_model(self.model.llama_model), "disable_adapter"):
                    raise ValueError(
                        "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                    )
            elif hasattr(model, "llm_model"):
                if not hasattr(self.accelerator.unwrap_model(self.model.llm_model), "disable_adapter"):
                    raise ValueError(
                        "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
                    )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")


    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):

        import deepspeed
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:


                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )


        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model


    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
        rates: Optional[torch.FloatTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        "Compute the DPO loss for a batch of policy and reference model log probabilities."
        device = self.accelerator.device
        policy_logratios = (policy_chosen_logps - policy_rejected_logps).to(device)
        ref_logratios = (reference_chosen_logps - reference_rejected_logps).to(device)

        logits = policy_logratios - ref_logratios

        losses = -F.logsigmoid(self.dpo_beta * logits)


        chosen_rewards = self.dpo_beta * (policy_chosen_logps.to(device) - reference_chosen_logps.to(device)).detach()
        rejected_rewards = (
            self.dpo_beta * (policy_rejected_logps.to(device) - reference_rejected_logps.to(device)).detach()
        )

        return losses, chosen_rewards, rejected_rewards


    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        consider_first_token_to_eos: bool = False,
    ) -> torch.FloatTensor:
        "Compute the log probabilities of the given labels under the given logits."
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not self.is_encoder_decoder:
            # Shift so that tokens < n predict n
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id


        labels[labels == self.label_pad_token_id] = 0


        per_token_logps = torch.zeros_like(labels, dtype=torch.float32)
        per_token_logps[loss_mask] = -F.cross_entropy(logits[loss_mask].float(), labels[loss_mask], reduction="none")

        masked_per_token_logps = per_token_logps * loss_mask

        if consider_first_token_to_eos:

            first_non_pad_indices = (loss_mask.cumsum(dim=1) == 1).nonzero(as_tuple=True)

            eos_token_logps = logits.log_softmax(-1)[:, :, self.processing_class.eos_token_id]
            first_token_to_eos_logps = eos_token_logps[first_non_pad_indices]


            half_batch_size: int = logits.shape[0] // 2
            second_half_indices = torch.arange(half_batch_size, logits.shape[0], device=logits.device)


            for idx in second_half_indices:

                seq_index = first_non_pad_indices[1][first_non_pad_indices[0] == idx][0]

                avg_logp = (masked_per_token_logps[idx, seq_index] + first_token_to_eos_logps[idx]) / 2

                masked_per_token_logps[idx, seq_index] = avg_logp

        if average_log_prob:
            return masked_per_token_logps.sum(-1) / loss_mask.sum(-1)
        else:
            return masked_per_token_logps.sum(-1)

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
    ):
        return None, None, None

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        "Overriding built-in evaluation loop to store metrics for each batch."


        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        "Log `logs` on the various objects watching training, including stored metrics."

        train_eval = "train" if "loss" in logs else "eval"

        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super().log(logs, start_time)
        else:
            return super().log(logs)

    def dpo_concatenated_forward(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        raise NotImplementedError

    def get_batch_loss_metrics(**kwargs):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        raise NotImplementedError

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
        raise NotImplementedError
