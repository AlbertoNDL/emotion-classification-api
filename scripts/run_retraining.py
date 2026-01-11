from src.incremental_retrain import retrain

if __name__ == "__main__":
    retrain(
        base_model_path="models/emotions-v1",
        calibration_data_path="data/emotions_augmented.csv",
        output_dir="models/emotions-v1.1",
        epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )