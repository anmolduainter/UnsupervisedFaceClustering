import openvino as ov
import torch
import onnx
from backbones.mobilefacenet import MobileFaceNet
from backbones.mobilefacenetv2 import MobileFaceNetv2

import nncf
from dataset import FaceDataset
import openvino as ov
from nncf import compress_weights, CompressWeightsMode
import os

def MakeDir(p):
    if (not os.path.exists(p)):
        os.mkdir(p)

def load_model(model_type, model_path):
    if (model_type == "mobilefacenet"):
            backbone=MobileFaceNet()
    elif (model_type == "mobilefacenetv2"):
            backbone=MobileFaceNetv2()

    backbone.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    backbone.eval()
    return backbone

def create_and_save_onnx(model, dummy_input, model_type, root_path):
    MakeDir(root_path + model_type + "/onnx/")
    torch.onnx.export(
        model,
        dummy_input,       # model input (or a tuple for multiple inputs) 
        root_path + model_type + "/onnx/" + "optim.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=10,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
    )
    print ("Saved Onnx")

def create_and_save_in_ir(root_path, model_type):
    ov_model = ov.convert_model(root_path + model_type + "/onnx/" + "optim.onnx")
    sp = root_path + model_type + "/openvino_optimized/fp16/"
    # ov_model = compress_weights(ov_model, mode=CompressWeightsMode.INT4_SYM, group_size=128, ratio=0.99) # model is openvino.Model object
    ov.save_model(ov_model, sp + "optim.xml")
    print ("Saved FP16 model")

def post_training_quantization(root_path, model_type):
    print ("==== Doing PTQ ====")
    rec = "/data/Kaggle/FaceClust/Datasets/"
    train_l = [
        "105_classes_pins_dataset_train",
        "CASIA",
        "CelebaTrain",
        "DigiFace"
    ]

    calibration_loader = torch.utils.data.DataLoader(
        FaceDataset(root_dir=rec, folder_l = train_l)
    )

    def transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
    model = ov.Core().read_model(root_path + model_type + "/openvino_optimized/fp16/" + "optim.xml")
    quantized_model = nncf.quantize(model, calibration_dataset, target_device=nncf.TargetDevice.CPU)
    ov.save_model(quantized_model, root_path + model_type + "/openvino_optimized/ptq/" + "quantized_model.xml")

    print ("==== Done ====")

root_path = "./model_weights/"
print ("==== Optimizing MobileFaceNet =======")
model_type = "mobilefacenet"
model_path = root_path + model_type + "/fp32/97272backbone.pth"
dummy_input = torch.randn((1,3,112,112), requires_grad=True)

model = load_model(model_type, model_path)
create_and_save_onnx(model, dummy_input, model_type, root_path)
create_and_save_in_ir(root_path, model_type)
post_training_quantization(root_path, model_type)

print ("Done!")

print ("===== Optimizing MobileFaceNetv2 =======")
model_type = "mobilefacenetv2"
model_path = root_path + model_type + "/fp32/29722backbone.pth"
dummy_input = torch.randn((1,3,112,112), requires_grad=True)

model = load_model(model_type, model_path)
create_and_save_onnx(model, dummy_input, model_type, root_path)
create_and_save_in_ir(root_path, model_type)
post_training_quantization(root_path, model_type)

print ("Done!")
