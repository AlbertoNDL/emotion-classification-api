from src.transform import convert_to_onnx

convert_to_onnx(
    model_dir="models/emotions-v1.1",
    onnx_path="models/onnx/model.onnx",
    max_length=128
)
