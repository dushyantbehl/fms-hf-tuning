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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union, Dict
import logging
import os
import json

from tuning.utils.utils import get_extension, load_yaml_or_json

from datasets import Dataset, IterableDataset, interleave_datasets
from transformers import Autotokenizer

logger = logging.getLogger(__name__)

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

def _validate_data_handler_config(data_handler) -> DataHandlerConfig:
    kwargs = data_handler
    assert isinstance(kwargs, dict), "data_handlers in data_config needs to be a dict"
    assert "name" in kwargs and isinstance(kwargs['name'], str), "data_handlers need to have a name with type str"
    assert "arguments" in kwargs, "data handlers need to have arguments"
    assert isinstance(kwargs['arguments'], dict), "data handler arguments should be of the type dict"
    return DataHandlerConfig(**kwargs)

def _validate_dataset_config(dataset_config) -> DatasetConfig:
    c = DatasetConfig()
    kwargs = dataset_config
    assert isinstance(kwargs, dict), "dataset_config in data_config needs to be a dict"
    if "name" in kwargs:
        assert isinstance(kwargs["name"], str), "dataset name should be string"
        c.name = kwargs['name']
    if "data_paths" not in kwargs:
        raise ValueError("data_paths should be specified for each dataset")
    else:
        data_paths = kwargs['data_paths']
        assert(isinstance(data_paths, List), "data_paths should be an array of files")
        c.data_paths = []
        for p in data_paths:
            assert isinstance(p, str), f"path {p} should be of the type string"
            assert os.path.exists(p), f"data_paths {p} does not exist"
            if not os.isabs(p):
                _p = os.path.abspath(p)
                logger.warning(f' Provided path {p} is not absolute changing it to {_p}')
                p = _p
            c.data_paths.append(p)
    if "sampling" in kwargs:
        sampling_kwargs = kwargs['sampling']
        assert isinstance(Dict, sampling_kwargs), "sampling arguments should be of the type dict"
        if "ratio" in sampling_kwargs:
            ratio = sampling_kwargs['ratio']
            assert((isinstance(ratio, float) and (0 <= ratio <= 1.0)), 
               f"sampling ratio: {ratio} should be float and in range [0.0,1.0]")
        c.sampling = sampling_kwargs
    if "splitter_arguments" in kwargs:
        splitter_kwargs = kwargs['splitter_arguments']
        assert isinstance(Dict, splitter_kwargs), "splitter_arguments should be of the type dict"
        c.splitter_arguments = splitter_kwargs
    if "data_handlers" in kwargs:
        c.data_handlers = []
        for handler in kwargs['data_handlers']:
            c.data_handlers.append(_validate_data_handler_config(handler))
    return c

def _validate_dataloader_config(dataloader_config) -> DataLoaderConfig:
    kwargs = dataloader_config
    c = DataLoaderConfig()
    assert isinstance(kwargs, dict), "dataloader in data_config needs to be a dict"
    if "streaming" in kwargs:
        assert (isinstance(kwargs['streaming'], bool), 
                "streaming should be a boolean true or false")
        c.streaming = kwargs['streaming']
    return c

def load_and_validate_data_config(data_config_file: str) -> DataConfig:
    raw_data = load_yaml_or_json(data_config_file)
    assert isinstance(raw_data, Dict), f"The provided data_config file is invalid: {data_config_file}"
    data_config = DataConfig()
    assert "datasets" in raw_data, "datasets should be provided in data config"
    assert isinstance(raw_data['datasets'], List), "datasets should be provided as a list"
    data_config.datasets = []
    for d in raw_data['datasets']:
        data_config.datasets.append(_validate_dataset_config(d))
    if "dataloader" in data_config:
        dataloader = _validate_dataloader_config(raw_data['dataloader'])
        data_config.dataloader = dataloader
    return data_config

class HFJSONDataHandler:

    data_files: List[str]
    input_field_name: str
    output_field_name: str

    def __init__(self, data_files: List[str], input_field_name: str, output_field_name: str):
        self.data_files = data_files
        self.input_field_name = input_field_name
        self.output_field_name = output_field_name

    # This will load the files one by one.
    # TODO: Do we implement shuffling here? Per file? Across files?
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

class HFParquetDataHandler:

    data_files: List[str]

    def __init__(self, data_files):
        self.data_files = data_files
        pass

    def load_dataset(self):
        pass

class HFArrowDataHandler:

    data_files: List[str]

    def __init__(self, data_files):
        self.data_files = data_files
        pass

    def load_dataset(self):
        pass

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

class DataLoader:

    tokenizer = None
    model_name_or_path = None
    block_size = None
    data_config: DataConfig = None

    def __init__(self, dataconfig: DataConfig, tokenizer, model_name_or_path, block_size):
        self.data_config = dataconfig
        self.tokenizer = tokenizer
        self.model_name_or_path = model_name_or_path
        self.block_size = block_size

    @abstractmethod
    def process_data_config(data_config: DataConfig):
        pass

class StatefulDistributedDataLoader(DataLoader):

    def __init__(self, dataconfig: DataConfig, tokenizer, model_name_or_path, block_size):
        super.__init__(dataconfig, tokenizer, model_name_or_path, block_size)

    def process_data_config(data_config: DataConfig):
        # TODO: Not implemented.
        pass

# This is the non distributed function. 
# For the distributed dataset we need to see how to distribute
# pre tokenization and post tokenization with the dataset.
class HFBasedDataLoader(DataLoader):

    def __init__(self, dataconfig: DataConfig, tokenizer, model_name_or_path, block_size):
        super.__init__(dataconfig, tokenizer, model_name_or_path, block_size)

    # data_config = load_and_validate_data_config(data_config_file)
    # logger.info("Starting data loading...")
    def process_data_config(self):

        train_datasets = None
        eval_datasets = None
        test_datasets = None

        # TODO: We need to get the accelerator
        with accelerator.main_process_first(desc="Processing data..."):

            for d in self.data_config.datasets:
                logger.info("Loading %s" % (d.name))
                data_loader = load_files_by_type(data_files=d.data_paths)
                raw_dataset = data_loader.load_dataset()

                if isinstance(raw_dataset, datasets.IterableDataset):
                    raw_datasets = datasets.IterableDatasetDict()
                    if eval_datasets == None:
                        eval_datasets = datasets.IterableDatasetDict()
                    if test_datasets == None:
                        test_datasets = datasets.IterableDatasetDict()
                else:
                    raw_datasets = datasets.DatasetDict()
                    if eval_datasets == None:
                        eval_datasets = datasets.DatasetDict()
                    if test_datasets == None:
                        test_datasets = datasets.DatasetDict()

                splitName = "train" # default
                # Assume all is train split, if splitter is requested this will change
                raw_datasets[splitName] = raw_dataset

                if d.splitter_arguments:
                    kwargs = d.splitter_arguments
                    raw_datasets = raw_dataset.train_test_split(**kwargs)

                    # TODO: Why are we doing this?
                    raw_datasets["eval"] = raw_datasets["test"]
                    raw_datasets.pop("test")

                if d.data_handlers:
                    for data_handler in d.data_handlers:
                        kwargs = data_handler.arguments
                        handler_name = data_handler.name
                        if "batched" not in kwargs:
                            kwargs["batched"] = True

                        column_names = raw_datasets[splitName].column_names
                        # remove __content__ from all processing
                        if "__content__" in column_names:
                            column_names.remove("__content__")
                        if "remove_columns" not in kwargs:
                            kwargs["remove_columns"] = None
                        if kwargs["remove_columns"] == "all":
                            kwargs["remove_columns"] = column_names
                        
                        if "fn_kwargs" not in kwargs:
                            kwargs["fn_kwargs"] = {}

                        if isinstance(raw_datasets, datasets.DatasetDict) and "num_proc" not in kwargs:
                            kwargs["num_proc"]=os.cpu_count()

                        # TODO: Need to expose these.
                        kwargs["fn_kwargs"]["tokenizer"] = self.tokenizer
                        kwargs["fn_kwargs"]["block_size"] = self.fm_args.block_size
                        kwargs["fn_kwargs"]["column_names"] = column_names
                        kwargs["fn_kwargs"]["model_path"] = self.fm_args.base_model_path

                        logger.info("Loaded raw dataset : {raw_datasets}")

                        # TODO: Where are the registered handlers?
                        #       Where are the user provided ones?
                        raw_datasets=raw_datasets.map(self.data_handlers[handler_name], **kwargs)

                        # TODO: What does this code do?
                        #if name == "preprocess_prompt":
                        #    self.task = kwargs["fn_kwargs"]["task"]

                if "eval" in raw_datasets:
                    eval_datasets["eval_" + d.name] = raw_datasets["eval"]
                if "test" in raw_datasets:
                    test_datasets["test_" + d.name] = raw_datasets["test"]
                
                if dataset_with_splits is None:
                    dataset_with_splits = raw_datasets
                else:
                    for k in raw_datasets.keys():
                        if k in dataset_with_splits:
                            dataset_with_splits[k] = datasets.concatenate_datasets([dataset_with_splits[k], raw_datasets[k]])
                        else:
                            dataset_with_splits[k] = raw_datasets[k]

        if "train" in dataset_with_splits:
            train_datasets = dataset_with_splits["train"]
        
        if "eval" in dataset_with_splits:
            if len(eval_datasets) > 1:
                eval_datasets["eval"] = dataset_with_splits["eval"]
            else:
                eval_datasets = dataset_with_splits["eval"]
        
        if "test" in dataset_with_splits:
            if len(test_datasets) > 1:
                test_datasets["test"] = dataset_with_splits["test"]
            else:
                test_datasets = dataset_with_splits["test"]
        
        if len(eval_datasets) == 0:
            eval_datasets = None 
        if len(test_datasets) == 0:
            test_datasets = None

        return (train_datasets, test_datasets, eval_datasets)