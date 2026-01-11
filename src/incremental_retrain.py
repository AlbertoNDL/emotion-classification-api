import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from src.config import TrainingConfig

def retrain(
    base_model_path: str,
    calibration_data_path: str,
    output_dir: str,
    labels: list = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    seed: int = None
):
    cfg = TrainingConfig()
    labels = labels or cfg.labels
    epochs = epochs or cfg.num_train_epochs
    batch_size = batch_size or cfg.per_device_train_batch_size
    learning_rate = learning_rate or cfg.learning_rate
    seed = seed or cfg.seed

    df = pd.read_csv(calibration_data_path, sep=";")
    df["label"] = df["label"].replace({"love": "joy"})
    
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["labels"] = df["label"].map(label2id)

    dataset = Dataset.from_pandas(df[["text", "labels"]])
    dataset = dataset.train_test_split(test_size=0.1, seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model_config = AutoConfig.from_pretrained(
        base_model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    print("Loaded model config:", model_config)
    print("Labels:", labels)
    print("Label2ID:", label2id)
    print("ID2Label:", id2label)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        config=model_config,
        ignore_mismatched_sizes=True
    )

    for param in model.base_model.parameters():
        param.requires_grad = False

    label_counts = df["labels"].value_counts().sort_index()
    weights = 1.0 / label_counts

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=cfg.load_best_model_at_end,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer
    )

    trainer.train()

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    return {
        "num_labels": len(labels),
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "model_path": output_dir
    }


if __name__ == "__main__":
    retrain(
        base_model_path="models/emotions-v1",
        calibration_data_path="data/emotions_augmented.csv",
        output_dir="models/emotions-v1.1",
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
