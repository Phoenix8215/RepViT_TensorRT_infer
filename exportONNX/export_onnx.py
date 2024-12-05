import torch
import time
from timm import create_model
import model
import torch
import torch.onnx
import onnx
import onnxsim
def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)

model = create_model("repvit_m2_3", num_classes=1000)


checkpoint = torch.load('repvit_m2_3_distill_450e.pth', map_location=torch.device('cpu'))

state_dict = checkpoint['model']

model.load_state_dict(state_dict, strict=False)

# 结构化重参
replace_batchnorm(model)


model.eval()

dummy_input = torch.randn(1, 3, 224, 224)  # 根据您的模型调整

onnx_file_path = "onnx/repvit_m2_3_distill_450e.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_file_path,
    export_params=True,
    opset_version=15,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
)

print(f"模型已成功转换为ONNX格式，保存在 {onnx_file_path}")

onnx_model = onnx.load(onnx_file_path)
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX模型结构验证通过！")
except onnx.checker.ValidationError as e:
    print("ONNX模型验证失败：", e)


