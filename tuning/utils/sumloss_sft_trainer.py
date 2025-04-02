import torch
from trl import SFTTrainer
from typing import override
from transformers.utils import logging

logger = logging.get_logger(__name__)

class SumLossSFTTrainer(SFTTrainer):

    embedding_size: int
    _total_train_tokens: float

    def __init__(
        self,
        embedding_size: int,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_size = embedding_size
        self._total_train_tokens = 0
        logger.warning(f"===> Initialized SumLossSFTTrainer with embedding size {embedding_size}")
        # Disable model loss kwargs as we are overriding the model loss
        self.model_accepts_loss_kwargs = False

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #def compute_loss_func(outputs, labels, num_items_in_batch):
        """
        Function to switch loss function calculation to reduce_loss=sum
        """
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits
        labels = inputs["labels"]

        # reduce loss is sum
        # this ensures that we weight all tokens in the dataset equally,
        # rather than weighting each overall example equally when
        # using high amounts of gradient accumulation.
        # this can result in > 5 point improvements in AlpacaEval
        # see https://github.com/huggingface/transformers/issues/24725 for
        # more discussion and details.

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # get the loss function
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")

        # Flatten tensors
        shift_logits = shift_logits.view(-1, self.embedding_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    
        return (loss, outputs) if return_outputs else loss