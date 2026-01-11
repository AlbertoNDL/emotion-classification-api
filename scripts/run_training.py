from src.train import train

if __name__ == "__main__":
    train(
        data_path="data/emotions.csv",
        model_path="j-hartmann/emotion-english-distilroberta-base",
        output_dir="models/emotions-v1",
        epochs=3,
        batch_size=16
    )
