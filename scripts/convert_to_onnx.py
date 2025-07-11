import torch
import torch.onnx
from models import LightweightUNet  # Make sure this matches your actual model name

def export_to_onnx(model_path="student_model_best.pth", onnx_path="student_model.onnx", input_shape=(1, 3, 128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LightweightUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_shape).to(device)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )

    print(f"✅ ONNX model exported successfully to: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
