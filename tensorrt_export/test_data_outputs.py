import os
import torch
import onnxruntime as ort
from polygraphy.comparator import RunResults
from polygraphy.json import save_json


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
new_onnx_dir = os.path.join(output_dir, "onnx_output_no_cache_new")
new_onnx_path = os.path.join(new_onnx_dir, "chatglm2_6b.onnx")
input_path1 = os.path.join(output_dir, "pt_input1.pt")
output_path1 = os.path.join(output_dir, "pt_output1.pt")
input_dict = torch.jit.load(input_path1)
batch_size = 1
num_layers = 1
output_dict = torch.jit.load(output_path1)
input_ids = input_dict.input_ids.numpy()
position_ids = input_dict.position_ids.numpy()
session = ort.InferenceSession(new_onnx_path, providers=["CPUExecutionProvider"])
outputs = session.get_outputs()
output_names = [out.name for out in outputs]
outputs = session.run(output_names, {"input_ids": input_ids, "position_ids": position_ids})
print(len(outputs))
print(type(outputs))
output_dict = {name: value for (name, value) in zip(output_names, outputs)}
for k, v in output_dict.items():
    if v is None:
        print(k, "is None")
custom_outputs = RunResults()
custom_outputs.add(out_list=[output_dict], runner_name="onnxrt")
custom_outputs.save("custom_outputs.json")


