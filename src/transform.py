import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

def convert_to_onnx(model_dir: str, onnx_path: str, max_length: int = 128):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.eval()
    dummy_input = tokenizer(
        ["This is a test sentence."],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"}
        },
        opset_version=17,
    )

    print(f"ONNX model saved to {onnx_path}")