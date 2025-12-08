import torch
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/vit-base-patch16-224-in21k"
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, device=device)
onnx_path = "model.onnx"

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=20,
        do_constant_folding=True,
    )
