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

# Third Party
from transformers.utils import logging
from trl import SFTTrainer
import torch
from torch import nn
from typing import Union, Any
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)
logger = logging.get_logger(__name__)

DEFAULT_LABELS_KEY = "labels"

################### Some Notes on the below loss calculation. ##############################
# This loss function just replaces trainer loss function to calculate sum reduction
# over the model forward pass. This function is useful while using high amount of
# gradient accumulation to ensure all tokens are accounted for equally.
#
# In HF Trainer performing a *true* reduce loss sum calculation requires to change
# the trainier.training_step function as well to ensure the `backwards` call is done
# on the combined `sum` loss rather than a loss which is scaled with respect to GAS.
# See these lines - https://github.com/huggingface/transformers/blob/\
#                           08e3217bafddc5d11ce0e7369bcfaaabe5501ba5/\
#                           src/transformers/trainer.py#L3765C1-L3774C54
#
# The methodology we have is not too intrusive and not to change `training_step()` as
# that is equivalent of almost recreating a training loop. Our approach is more towards
# providing a close approximation to the reduce loss sum calculation.
#
# This feature is provided as an experimental feature and not fully claimed to be supported.
#
# Known limitation (will be fixed in the next releases) -
#   Not fully tested and compatible with PEFT especially PEFT PT.
#############################################################################################


class SumLossSFTTrainer(SFTTrainer):

    vocab_size: int

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        vocab = self.model.get_input_embeddings()
        self.vocab_size = vocab.weight.shape[0]

        # Disable model loss kwargs as we are overriding the model loss
        # This is so that the loss calculated by us is divided over actual
        # actual gradient accumulation steps inside HF Trainer
        #
        # See this code -
        # https://github.com/huggingface/transformers/blob/\
        #      41b9b92b52215bed472c9a534a06abbc3a9a95cd/src/transformers/trainer.py#L3769
        self.model_accepts_loss_kwargs = False
        logger.info(
            "✅ Initialized SumLossSFTTrainer. "
            + "Switching trainer loss function with cross entropy loss sum reduction.\n"
            + "This is an experimental feature and should be used as such "
            + " please report any issue you see with this function to the maintainers"
        )

    def training_step(
            self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
        ) -> torch.Tensor:
            """
            Function overridden from SFTTrainer Perform a training step on a batch of inputs.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to train.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.

            Return:
                `torch.Tensor`: The tensor with training loss on this batch.
            """
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available(min_version="2.0"):
                    torch.mps.empty_cache()
                elif is_torch_hpu_available():
                    logger.warning(
                        "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                    )
                else:
                    torch.cuda.empty_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learnign rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            self.accelerator.backward(loss, **kwargs)

            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            return loss.detach()

    # Overrides trl/sft_trainer::SFTTrainer compute_loss function.
    #
    # This loss function is taken from OpenInstruct
    #
    # https://github.com/allenai/open-instruct/blob/open_instruct/finetune.py
    #
    # Using sum reduction for CrossEntropyLoss according to OpenInstruct
    # helps in ensuring all tokens are weighed equally in the dataset which is
    # important for high amount of gradient accumulation steps.
    # For more details see their discussion on this transformers issue
    # URL - https://github.com/huggingface/transformers/issues/24725
    #
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Function to switch loss function calculation to reduce_loss=sum
        """
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        # Run the forward pass
        outputs = model(**inputs, use_cache=False)

        # Extract logits and perform calculation for loss outside the modelling class
        logits = outputs.logits
        labels = inputs[DEFAULT_LABELS_KEY]

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # get the loss function from torch with sum reduction
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

        # Flatten tensors as expected by crossentropyloss
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Shift the data to device and run loss.
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss
