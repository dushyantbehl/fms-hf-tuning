# Standard
import re


def is_map_batched(element):
    # This `element` at the very least is a dict of values
    # or for batched operation a dict with values as array.
    return isinstance(next(iter(element.values())), list)

def custom_data_formatter(element, template, formatted_dataset_field):
    def replace_text(match_obj, element):
        captured_groups = match_obj.groups()
        if len(captured_groups) != 1:
            raise ValueError(
                "Unexpectedly captured multiple groups in template formatting"
            )

        index_object = captured_groups[0]
        if index_object not in element:
            raise KeyError("Requested template string is not a valid key in dict")

        return element[index_object]

    def formatted_item(element):
        return {
            formatted_dataset_field: re.sub(
                r"{{([\s0-9a-zA-Z_\-\.]+)}}",
                lambda match: replace_text(match, element),
                template,
            )
        }

    if not is_map_batched(element):
        return formatted_item(element)

    batch_size = len(next(iter(element.values())))
    formatted_batch = []
    for i in range(batch_size):
        formatted_batch.append(
            formatted_item({key: element[key][i] for key in element})
        )
    return {
        formatted_dataset_field: [b[formatted_dataset_field] for b in formatted_batch]
    }


def apply_custom_formatting_template(
    dataset, template, formatted_dataset_field, eos_token=""
):
    """Function to format datasets with Alpaca style / other templates.
    Args:
        dataset: the HF Dataset element loaded from a JSON or DatasetDict object.
        template: Template to format data with. Features of Dataset
            should be referred to by {{key}}
        formatted_dataset_field: Dataset_text_field
        eos_token: string EOS token to be appended while formatting data to a single sequence.
            Defaults to empty
    Returns:
        Formatted HF Dataset
    """

    template += eos_token

    if not formatted_dataset_field:
        raise ValueError(
            "Unable to apply custom formatting because the formatted_dataset_field was not provided"
        )

    fn_kwargs = {}
    fn_kwargs["template"] = template
    fn_kwargs["formatted_dataset_field"] = formatted_dataset_field

    return dataset.map(custom_data_formatter, fn_kwargs=fn_kwargs)
