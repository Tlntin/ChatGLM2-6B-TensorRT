import onnxruntime as ort
import torch
import numpy as np
import os
from colored import fg, stylize


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
# model_dir = os.path.join(project_dir, "chatglm_6b")
onnx_path = os.path.join(output_dir, "onnx_output", "chatglm_6b.onnx")
new_onnx_dir = os.path.join(project_dir, "output", "new_onnx_output")
if not os.path.exists(new_onnx_dir):
    os.mkdir(new_onnx_dir)
new_onnx_path = os.path.join(new_onnx_dir, "chatglm_6b.onnx")


def compare_value(pre_numpy: np.array, true_numpy: np.array):
    assert pre_numpy.shape == true_numpy.shape
    diff = np.abs(pre_numpy - true_numpy).max()
    if diff > 1e-3:
        print(stylize(f"diff: {diff} is_pass: failed", fg("red")))
    else:
        print(stylize(f"diff: {diff} is_pass: OK", fg("green")))
    return diff

# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
providers = [("CUDAExecutionProvider", {'enable_cuda_graph': False})]
sess_options = ort.SessionOptions()
sess_options.optimized_model_filepath = new_onnx_path
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    onnx_path, sess_options=sess_options, providers=providers
)
print(session.get_providers())

# cuda device id
device_id = 0
input_path1 = os.path.join(output_dir, "pt_input1.pt")
output_path1 = os.path.join(output_dir, "pt_output1.pt")
input_dict = torch.jit.load(input_path1)
output_dict = torch.jit.load(output_path1)
input_ids = input_dict.input_ids.data.cpu().numpy().astype(np.int64)
position_ids = input_dict.position_ids.data.cpu().numpy().astype(np.int64)
attention_mask = input_dict.attention_mask.data.cpu().numpy()
logits = output_dict.logits.data.cpu().numpy()
key = "present_key_values.0.decorder.key"
one_present_key = getattr(output_dict, key).data.cpu().numpy()

io_binding = session.io_binding()
io_binding.bind_ortvalue_input(
    "input_ids",
    ort.OrtValue.ortvalue_from_numpy(input_ids, "cuda", device_id=device_id)
)
io_binding.bind_ortvalue_input(
    "position_ids",
    ort.OrtValue.ortvalue_from_numpy(position_ids, "cuda", device_id=device_id)
)
io_binding.bind_ortvalue_input(
    "attention_mask",
    ort.OrtValue.ortvalue_from_numpy(attention_mask, "cuda", device_id=device_id)
)
for layer_idx in range(28):
    input_names = [
        f"past_key_values.{layer_idx}.decorder.key",
        f"past_key_values.{layer_idx}.decorder.value"
    ]
    # inputs[input_names[0]] = past_key_values
    # inputs[input_names[1]] = past_key_values
    for name in input_names:
        past_key_values = np.zeros([0, 1, 32, 128], dtype=one_present_key.dtype)
        io_binding.bind_ortvalue_input(
            name=name,
            ortvalue=ort.OrtValue.ortvalue_from_numpy(
                past_key_values, "cuda", device_id=device_id
            )
        )
    output_name = [
        f"present_key_values.{layer_idx}.decorder.key",
        f"present_key_values.{layer_idx}.decorder.value"
    ]
    for name in output_name:
        output_value = np.zeros(
            [input_ids.shape[1], 1, 32, 128],
            dtype=one_present_key.dtype
        )
        io_binding.bind_ortvalue_output(
            name=name,
            ortvalue=ort.OrtValue.ortvalue_from_numpy(
                output_value, "cuda", device_id=device_id
            )
        )
logits_numpy = np.zeros([1, input_ids.shape[1], 130528], dtype=logits.dtype)
io_binding.bind_ortvalue_output(
    name="logits",
    ortvalue=ort.OrtValue.ortvalue_from_numpy(
        logits_numpy
    )
)

# print(inputs)
session.run_with_iobinding(io_binding)
# compile logists
print('=' * 20)
print("compare logits")
pred_outputs = io_binding.copy_outputs_to_cpu()
compare_value(pred_outputs[-1], logits)

# compile present_key_values
for i in range(28):
    key_name = f"present_key_values.{i}.decorder.key"
    value_name = f"present_key_values.{i}.decorder.value"
    print('=' * 20)
    print(f"compare {key_name}")
    # key_numpy = [key_name]
    key_true = getattr(output_dict, key_name).data.cpu().numpy()
    key_pred = pred_outputs[i * 2]
    compare_value(key_pred, key_true)
    print('=' * 20)
    print(f"compare {value_name}")
    value_pred = pred_outputs[i * 2 + 1]
    value_true = getattr(output_dict, value_name).data.cpu().numpy()
    compare_value(value_pred, value_true)


