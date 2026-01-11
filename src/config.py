from dataclasses import dataclass
from typing import List

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    num_labels: int = 6
    labels: List[str] = (
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutral",
        "sadness",
        "surprise",
    )

    # Data
    text_column: str = "text"
    label_column: str = "label"

    # Training
    output_dir: str = "artifacts/model"
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    weight_decay: float = 0.01
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 50
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"

    # Misc
    seed: int = 42
