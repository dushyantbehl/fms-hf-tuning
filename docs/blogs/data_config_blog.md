# Welcome to Easy, Powerful Data Preprocessing with `fms-hf-tuning`

If you're working on fine-tuning foundation models, you know the importance of flexible, reliable data pipelines. That’s exactly what this library delivers—with **simplicity** at its core.

This guide will walk you through how this library helps you structure, transform, and prepare your datasets using nothing more than a configuration file. Whether you're dealing with multiple datasets, need to split them intelligently, or want to stream them in chunks, this tool enables all of that and more.

---

## What You Can Do Without Writing a Single Line of Code

The `fms-hf-tuning` library was built to simplify the most common, often painful, preprocessing needs for model training:

- **Mix and match datasets of different formats** (e.g., `.jsonl`, folders of files).
- **Define sampling ratios** to control how much each dataset contributes.
- **Automatically split datasets** into training and validation subsets.
- **Stream massive datasets from disk or remote sources** without loading everything into memory.
- **Apply complex transformations** like filtering, renaming, and token manipulation using simple config declarations.

All of this is configured through a single YAML or JSON file—no Python scripts required.

---

## Quick Steps to Get Started

1. Install the `fms-hf-tuning` library.
2. Create a `data_config.yaml` using the examples in `tests/predefined_data_configs`.
3. Run:

```bash
python sft_trainer.py --data_config_path data_config.yaml <additional training arguments>
```

4. Sit back and let the tool handle preprocessing, mixing, and loading.

---

## What Does the `data_config` Look Like?

At the heart of the library is the `data_config` file, which defines every part of your preprocessing pipeline. Here’s what it typically includes:

### Top-Level `dataprocessor`

This section controls global options:

```yaml
dataprocessor:
  type: default
  streaming: true
  seed: 42
  sampling_stopping_strategy: all_exhausted
```

- `type`: Currently always set to `default`.
- `streaming`: Set to `true` to enable lazy loading.
- `seed`: Ensures reproducibility.
- `sampling_stopping_strategy`: Controls when to stop sampling. Options are `all_exhausted` or `first_exhausted`.

### Dataset Definitions

```yaml
datasets:
  - name: dataset_A
    data_paths:
      - "data/a.jsonl"
    sampling: 0.7

  - name: dataset_B
    data_paths:
      - "data/folder/"
    builder: custom_loader
    sampling: 0.3
```

Each dataset entry includes:

- `name`: A label for the dataset.
- `data_paths`: Path(s) to files or folders.
- `builder`: Optional custom loader function.
- `sampling`: A float (e.g., 0.3) that indicates the sampling weight.

If you specify sampling for more than one dataset, the library will use interleaved sampling based on the weights.

#### Mixing vs. Concatenation

By default, if you include multiple datasets without specifying `sampling`, they are simply concatenated together.

If you define `sampling` values for each dataset, the library switches to **interleaved sampling**, which randomly mixes examples from each dataset in proportion to their defined sampling weights. This is especially useful for balancing dataset contributions during training.

---

## Fine-Tuning Your Pipeline with Data Handlers

Need to manipulate your data before training? You can add `data_handlers`, which apply Hugging Face dataset operations like `map`, `filter`, `remove_columns`, and more:

```yaml
data_handlers:
  - name: remove_unused_columns
    arguments:
      remove_columns: "extra_meta"
      batched: false
      fn_kwargs: {}
```

Handlers are applied in the order you list them. Each one can be tailored with arguments that fit the specific transformation you're applying.

---

## Streaming Datasets at Scale

If you're working with huge datasets that can’t be loaded into memory, enable streaming mode:

```yaml
dataprocessor:
  streaming: true
```

Key things to keep in mind:

- Use `max_steps` instead of `num_train_epochs` in your trainer config.
- Streaming is **not compatible** with multipack plugins.

Streaming mode keeps memory usage low and allows you to handle datasets of arbitrary size efficiently.

---

## Built-In Splitting

If your dataset doesn’t already have a train/validation split, you can define one in the config:

```yaml
datasets:
  - name: fancy_data
    split:
      train: 0.8
      validation: 0.2
    data_paths:
      - data/fancy.jsonl
```

- For **non-streaming datasets**, you can specify any float split that adds up to 1.0.
- For **streaming datasets**, only full dataset splits (`1.0` or `0.0`) are supported.

Splits are applied before sampling so that your validation set remains untouched by any interleaving logic.

---

## How to Customize Chat Formatting

If you're working with chat-style datasets and want to change the format of prompts, you can include a `chat_template` in your config:

```yaml
datapreprocessor:
  chat_template: |
    {%- if messages[0]['role'] == 'system' %}
    ...
    {%- endif %}
```

You can define multi-line Jinja templates directly in your YAML file. This gives you full control over how conversational inputs are formatted for the model.

---

## Train and Validate—No Manual Steps

Once your `data_config.yaml` is ready, all you have to do is run the trainer script:

```bash
python sft_trainer.py --data_config_path data_config.yaml
```

The library will take care of:

- Loading the datasets
- Applying splits and handlers
- Mixing or streaming as configured

---

## Why You’ll Love It

| Feature                   | Benefit                              |
| ------------------------- | ------------------------------------ |
| **Zero-code setup**       | Define everything in YAML/JSON       |
| **Powerful mixing**       | Weighted sampling across datasets    |
| **Smart splitting**       | Clean train/val split automatically  |
| **Streaming support**     | Scale to large, disk-based corpora   |
| **Handler extensibility** | Customize pipelines easily           |
| **Prompt templating**     | Support for chat-style model formats |

---

## Final Thoughts

`fms-hf-tuning` makes advanced dataset preprocessing as simple as writing a configuration file. From dataset mixing and validation splitting to streaming and handler functions, it’s designed to eliminate boilerplate and empower researchers and practitioners to focus on model development.

If you're working with Hugging Face models or building custom training pipelines, this tool will save you time, reduce bugs, and standardize your workflows.

For detailed examples, check out the sample configurations in the repo.

**Happy fine-tuning!**

