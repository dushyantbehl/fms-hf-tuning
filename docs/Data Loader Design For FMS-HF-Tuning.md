T# Data Loader Design For FMS-HF-Tuning

**Deciders(s)**: Sukriti Sharma (sukriti.sharma4@ibm.com), Alexander Brooks (alex.brooks@ibm.com),  Raghu Ganti (rganti@us.ibm.com), Dushyant Behl (dushyantbehl@in.ibm.com), Ashok Pon Kumar (ashokponkumar@in.ibm.com)

**Date (YYYY-MM-DD)**:  2024-03-06

**Obsoletes ADRs**:  NA

**Modified By ADRs**:  NA

**Relevant Issues**: [1]

- [Summary and Objective](#summary-and-objective)
  - [Motivation](#motivation)
  - [User Benefit](#user-benefit)
- [Decision](#decision)
  - [Alternatives Considered](#alternatives-considered)
- [Consequences](#consequences)
- [Detailed Design](#detailed-design)

## Summary and Objective

The reason for motivating dataloader design for fms-hf-tuning is to have a unified interface which supports many type of data formats, streaming and non streaming data, weight based data mixing and many others. Full list we focus on is below - 

1. Data Loading -

    1. Different formats of data → Arrow, Parquet etc.

    1. Streaming data.

    1. Data Replay

    1. Resume with different number of GPUs

    1. Async data loading

1. Data Preprocessing - 

    1. Custom Attn Masking

    1. Tool Usage

1. Data Mixing - 

    1. Static weights based mixing

### Motivation

### User Benefit

## Decision

### Alternatives Considered

## Consequences

### Advantages

### Impact on performance

## Detailed Design

The input spec which user specifies on how to pass information to such data loader is this

```
dataloader:
    streaming: true
datasets:
  - name: dataset1
    sampling:
      ratio: 0.3
    splitter_arguments:
      test_size: 0
    data_paths:
      - /data/stackoverflow-kubectl_posts
      - /data/stackoverflow-kubernetes_posts
      - /data/stackoverflow-openshift_posts
    data_handlers:
      - name: render_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            jinja_template: "{<jinja-template>}"
  - name: dataset1
    sampling:
      ratio: 0.4
    data_paths:
      - /data/stackoverflow-kubectl_posts
      - /data/stackoverflow-kubernetes_posts
      - /data/stackoverflow-openshift_posts
    splitter_arguments:
      test_size: 1
    data_handlers:
      - name: render_template
        arguments:
          remove_columns: all
          batched: false
          fn_kwargs:
            jinja_template: "{<jinja-template>}"
  - name: dataset2
    sampling:
      ratio: 0.3
    data_handlers:
      - name: apply_tokenizer_chat_template
        arguments:
          remove_columns: all
          batched: false
    data_files:
      - /data/stackoverflow-kubectl_posts.jsonl
      - /data/stackoverflow-kubernetes_posts.jsonl
  - name: dataset2
    sampling:
      ratio: 0.3
    predefined_handlers:
      name: apply_chat_template # pretraining <tokenize and merge everything>
      fn_kwargs:
        jinja_template: "<>"
    data_files:
      - /data/stackoverflow-kubectl_posts.jsonl
      - /data/stackoverflow-kubernetes_posts.jsonl
```

Please note the various option differences in different type of datasets defined in the code. 
This is just a sample and is presented in YAML but the system is to be designed to take this input spec and parse it from JSON as well.

The config representation in code is specified below.

```

@dataclass
class DataHandlerConfig:
    name: str
    arguments: Optional[Dict]

@dataclass
class DatasetConfig:
    name: str
    sampling: Optional[Dict] = None
    splitter_arguments: Optional[Dict] = None
    data_paths: List[str]
    data_handlers: List[DataHandlerConfig] = None

@dataclass
class DataLoaderConfig:
    streaming: Optional[bool] = None

@dataclass
class DataConfig:
    dataloader: Optional[DataLoaderConfig]
    datasets: List[DatasetConfig]
```


The proposed design to implement support for this spec is follows,

```
class DataLoader:

    tokenizer = None
    model_name_or_path = None
    block_size = None
    data_config: DataConfig = None
    data_handlers: Dict[str, Callable] = None

    def __init__(self, dataconfig: DataConfig, tokenizer, model_name_or_path, block_size):
        self.data_config = dataconfig
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.block_size = block_size
        self.data_handlers = {}

    def register_data_handler(self, name: str, d: Callable):
        self.data_handlers[name] = d

    @abstractmethod
    def process_data_config(self, data_config: DataConfig):
        pass

    @abstractmethod
    def execute_data_handler(self, data_handler: DataHandlerConfig):
        pass
```

At the top level we propose to have this `class Dataloader` which is an abstract class 
and requires functions to process the data config proposed above.

We also propose a full length config verification code which preceeds the call to function 
`Dataloader.process_data_config` as the function expects a `DataConfig` object.

The data loader needs to support custom data handlers which are provided by users of the library
or even predefined handlers which need to be registered with the data loader class using the 
call `DataLoader.register_data_handler`.

The reason to have a top level dataloader class is needed to ensure we have separate classes for
hugging face and stateful data loader (fms-fsdp implementation).

As both data loaders behave separately we need to implement `Dataloader.execute_data_handler` function
to ensure data handlers are executed as per the dataloader apis before or after tokenization for example.

The process data loader goes through each `DataSetConfig` one by one and loads data files according to type
we need to implement loading code for different type of data formats like this.


```
def load_files_by_type(data_files, kwargs=None):
    extns = []
    for f in data_files:
        e = get_extension(f)
        extns.append(e)

    # simple check to make sure all files are of same type.
    # Do we need this assumption?
    assert (extns.count(extns[0]) == len(extns),
            "all files in a dataset should have same extension")

    e = extns[0]
    if e == "json" or e == "jsonl":
        return HFJSONDataHandler(data_files, **kwargs)
    elif e == "parquet":
        return HFParquetDataHandler(data_files)
    elif e == "arrow":
        return HFArrowDataHandler(data_files)
```

The individual handlers are implemented like this for example

```
class HFJSONDataHandler:

    data_files: List[str]
    input_field_name: str
    output_field_name: str

    def __init__(self, data_files: List[str], input_field_name: str, output_field_name: str):
        self.data_files = data_files
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name

    # This will load the files one by one.
    def __get_json_object(self):
        files = self.data_files
        for f in files:
            with open(f, "r", encoding="utf-8") as json_file:
                file_extension = get_extension(f)
                if file_extension == ".jsonl":
                    data_stream = (json.loads(line) for line in json_file)
                elif file_extension == ".json":
                    data_stream = json.load(json_file)
                else:
                    raise ValueError(f"JSONDataloader does not support {file_extension} format")

                for data in data_stream:
                    yield {
                        self.input_field_name: data[self.input_field_name],
                        self.output_field_name: data[self.output_field_name],
                    }

    def load_dataset(self):
        return Dataset.from_generator(self.__get_json_object)
```

The datatype handlers are not to be confused with other data handlers which are used to modify data samples before and after tokenization if needed.

The processing of the function is done by checking if a split is requested as per `test` or `train` dataset. 

If the top level dataloader has set `streaming` we need to pass that down to the datatype handlers to ensure
if it is supporter or not (like Json doesn't support it out of the box) while others support it but the argument needs to be passed down to the dataloader.

TODO: 
Add more details on how the stateful dataloader is to be implemented.
