---

# Welcome to Easy, Powerful Data Preprocessing with `fms-hf-tuning`

If you're working on fine-tuning foundation models, you know the importance of flexible, reliable data pipelines. That’s exactly what this library delivers—with **simplicity** at its core.

---

## What You Can Do Without Writing a Single Line of Code

- **Mix and match multiple datasets**—regardless of format.
- **Configure sampling ratios**, so each dataset appears in your training stream exactly how you want.
- **Split your data** into training and validation sets—all defined in a config file, no scripting needed.
- **Stream massive datasets chunk by chunk**, ideal when you can’t fit everything in memory.

Yep—**all with a simple YAML or JSON **``.

---

## What Does the `data_config` Look Like?

A `data_config` is a structured file (YAML or JSON) powered behind the scenes by a schema. Here's how it works:

At the top level, you define a ``, which specifies:

- `type` (currently always `"default"`)
- `streaming: true` if you want iterable streaming datasets
- A global random seed—for reproducibility
- Sampling stopping strategy: `"all_exhausted"` or `"first_exhausted"`
- Optional chat template overrides

Then list your **datasets** like this:

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

That simple config lets you mix A and B in a 70:30 ratio—no code, no hassle.

---

## Fine-Tuning Your Pipeline

The library includes "data handlers" you configure in the same file. Want to rename columns, remove tokens, filter examples? Just add handlers:

```yaml
data_handlers:
  - name: remove_unused_columns
    arguments:
      remove_columns: "extra_meta"
      batched: false
      fn_kwargs: {}
```

These run in order and can rely on Hugging Face’s `map`, `filter`, `rename`, etc.

---

## Mixing vs. Concatenation

- **Default**: All datasets are concatenated end-to-end.
- **With **``** defined**: The library switches to interleaving datasets with weights and strategies you choose. Simple and powerful.

---

## Built-In Splitting

Want train/validation splits without separate files? Add a `split:` section to any dataset:

```yaml
datasets:
  - name: fancy_data
    split:
      train: 0.8
      validation: 0.2
    data_paths:
      - data/fancy.jsonl
```

For streaming datasets (if `streaming: true`), only full splits (`1.0` or `0.0`) are supported. For non-streaming types, you’re free to choose any ratio less than or equal to 1.

---

## Train and Validate—No Manual Steps

- The library automatically splits your datasets.
- Sampling happens *after* splitting, so your validation set remains clean and consistent.
- You just run:

```bash
sft_trainer.py --data_config_path path/to/config.yaml
```

And you’re off to the model races.

---

## Want Streaming?

Set `streaming: true` in `dataprocessor` to load datasets lazily and avoid memory issues. Just remember:

- Use `max_steps`, not `num_train_epochs`, in your `TrainingArguments`
- Don’t use multipack plugin—it’s incompatible with streaming.

---

## How to Customize Chat Formatting

Need a custom prompt or chat template? Inject it into your config:

```yaml
datapreprocessor:
  chat_template: | 
    {%- if messages[0]['role'] == 'system' %}
    ...
    {%- endif %}
```

Grab the official template from Granite or define your own multiline version—and paste it straight into the `data_config`.

---

## Why You’ll Love It

| Feature                   | Benefit                             |
| ------------------------- | ----------------------------------- |
| **Zero-code setup**       | Define everything in YAML/JSON      |
| **Powerful mixing**       | Weighted sampling across datasets   |
| **Smart splitting**       | Clean train/val split automatically |
| **Streaming support**     | Scale to large, disk-based corpora  |
| **Handler extensibility** | Customize pipelines easily          |

---

## Quick Steps to Get Started

1. Install the library
2. Create a `data_config.yaml` (start from examples in `tests/predefined_data_configs`)
3. Run:

```bash
python sft_trainer.py --data_config_path data_config.yaml
```

4. Watch it automatically process, mix, split, and prepare your data—effortlessly.

---

## Final Thoughts

This library hides complexity behind a clean, declarative config file that even beginners can understand. Whether you’re ironing out a two-dataset mix, splitting your data, streaming terabytes of text, or just renaming columns—you don’t need to code it manually—just declare it and go.

If you’re fine-tuning foundation models and want a data pipeline that’s **flexible**, **robust**, and **super easy to use**, give `fms-hf-tuning` a go.

For detailed examples, check the sample configs—your next data pipeline is just a config away.

**Happy fine-tuning!**

