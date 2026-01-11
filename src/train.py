import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer

def train(data_path: str, model_path: str, output_dir: str, epochs: int = 3, batch_size: int = 16):
    df = pd.read_csv(data_path, sep=";")
    df["label"] = df["label"].replace({"love": "joy"})
    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["labels"] = df["label"].map(label2id)

    dataset = Dataset.from_pandas(df[["text", "labels"]])
    
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    config = AutoConfig.from_pretrained(
        model_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
        ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
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


