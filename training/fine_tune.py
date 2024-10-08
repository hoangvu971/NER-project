"""Fine-tune model and log to MLflow"""
import json
from dataclasses import dataclass, field
import os
from pathlib import Path
import dagshub
import mlflow
from mlflow.tracking import MlflowClient

import random
import sys
from typing import Optional

import torch
from gliner import GLiNER
from transformers import HfArgumentParser, set_seed
from gliner.data_processing.collator import DataCollator
from gliner.training import Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "true"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

# Set up MLflow and DagsHub
dagshub.init(repo_owner="hoangvu971", repo_name="NER-project", mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/hoangvu971/NER-project.mlflow")
mlflow.set_experiment("gliner_medium-v2.1")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=MODEL_DIR / "gliner_medium-v2.1",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: Optional[str] = field(
        default=DATA_DIR / "synthetic-pii-ner.json", metadata={"help": "Path to json dataset"}
    )
    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    with open(data_args.data_path, "r") as f:
        data = json.load(f)
    random.shuffle(data)
    train_dataset = data[: int(len(data) * 0.9)]
    test_dataset = data[int(len(data) * 0.9) :]

    model = GLiNER.from_pretrained(model_args.model_name_or_path)
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Log parameters to MLflow
    mlflow.log_params(
        {
            "model_name": model_args.model_name_or_path,
            "data_path": data_args.data_path,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "device": str(device),
            **training_args.to_dict(),
        }
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    mlflow.autolog()

    with mlflow.start_run():
        # Train the model
        trainer.train()

    print("Training completed. Results logged to MLflow on DagsHub.")


if __name__ == "__main__":
    main()
