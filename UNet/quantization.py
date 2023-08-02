import torch
from .model import Unet

model = Unet().cuda()

x = torch.randn(2, 3, 256, 256)
# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

######## ONNX Simplifier ############
import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load("model.onnx")

# convert model
model_simp, check = simplify(model)

########### Convert to FP16 ##########
from onnxmltools.utils.float16_converter import convert_float_to_float16
from onnxmltools.utils import save_model
new_onnx_model = convert_float_to_float16(model_simp)
save_model(new_onnx_model, 'model.fp16.onnx')