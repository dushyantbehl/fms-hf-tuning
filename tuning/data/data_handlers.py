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

# Definition of some predefined data preprocessing functions that we need.

# Standard
from typing import Dict, List

# Third Party
from transformers import AutoTokenizer

# Local
from tuning.utils.data_utils import custom_data_formatter, is_map_batched
from tuning.utils.preprocessing_utils import combine_sequence


def tokenize_and_apply_input_masking(
    element: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    input_field_name: str,
    output_field_name: str,
    **tokenizer_kwargs,
):
    def process(element):
        input = element[input_field_name]
        output = element[output_field_name]

        combined = combine_sequence(input, output, eos_token=tokenizer.eos_token)

        tokenized_comb_seqs = tokenizer(combined, **tokenizer_kwargs)
        tokenized_input = tokenizer(input, **tokenizer_kwargs)

        masked_labels = [-100] * len(
            tokenized_input.input_ids
        ) + tokenized_comb_seqs.input_ids[len(tokenized_input.input_ids) :]

        return {
            "input_ids": tokenized_comb_seqs.input_ids,
            "labels": masked_labels,
            "attention_mask": tokenized_comb_seqs.attention_mask,
        }

    if not is_map_batched(element):
        return process(element)

    batch_size = len(next(iter(element.values())))
    formatted_batch = []
    for i in range(batch_size):
        formatted_batch.append(process({key: element[key][i] for key in element}))
    return {key: [item[key] for item in formatted_batch] for key in formatted_batch[0]}


def apply_dataset_formatting(
    element: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    dataset_text_field: str,
    **kwargs,
):
    if not is_map_batched(element):
        return {
            f"{dataset_text_field}": element[f"{dataset_text_field}"]
            + tokenizer.eos_token
        }
    else:
        return {
            f"{dataset_text_field}": [
                text + tokenizer.eos_token for text in element[f"{dataset_text_field}"]
            ]
        }


def apply_custom_data_formatting_template(
    records: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    dataset_text_field: str,
    template: str,
    **kwargs,
):
    template += tokenizer.eos_token

    # TODO: Eventually move the code here.
    return custom_data_formatter(records, template, dataset_text_field)


AVAILABLE_DATA_HANDLERS = {
    "tokenize_and_apply_instruction_masking": tokenize_and_apply_input_masking,
    "apply_dataset_formatting": apply_dataset_formatting,
    "apply_custom_data_formatting_template": apply_custom_data_formatting_template,
}
