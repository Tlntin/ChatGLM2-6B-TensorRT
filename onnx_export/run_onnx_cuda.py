## 这个代码还有bug,尝试中，不要运行

import onnxruntime as ort
import numpy as np
import os


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
# model_dir = os.path.join(project_dir, "chatglm_6b")
onnx_path = os.path.join(project_dir, "output" ,"onnx_output", "chatglm_6b.onnx")
new_onnx_dir = os.path.join(project_dir, "output", "new_onnx_output")
if not os.path.exists(new_onnx_dir):
    os.mkdir(new_onnx_dir)
new_onnx_path = os.path.join(new_onnx_dir, "chatglm_6b.onnx")

# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
providers = [("CUDAExecutionProvider", {'enable_cuda_graph': True})]
sess_options = ort.SessionOptions()
sess_options.optimized_model_filepath = new_onnx_path
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    onnx_path, sess_options=sess_options, providers=providers
)
print(session.get_providers())

# cuda device id
device_id = 0
input_ids = np.array([[1, 5, 74874, 130001, 130004]], dtype=np.int64)
print("input_id.shape", input_ids.shape)
position_ids = np.array([[[0, 1, 2, 2, 2], [0, 0, 0, 1, 1]]], dtype=np.int64)
attention_mask = np.array(
    [[[
        [False, False, False, False, True],
        [False, False, False, False, True],
        [False, False, False, False, True],
        [False, False, False, False, True],
        [False, False, False, False, False]
    ]]], dtype=np.bool_
)

# nputs = {
#     "input_ids": input_ids,
#     "position_ids": position_ids,
#     "attention_mask": attention_mask,
# }
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
    past_key_values = np.zeros([0, 1, 32, 128], dtype=np.float16)
    # inputs[input_names[0]] = past_key_values
    # inputs[input_names[1]] = past_key_values
    for name in input_names:
        io_binding.bind_ortvalue_input(
            name=name,
            ortvalue=ort.OrtValue.ortvalue_from_numpy(
                past_key_values, "cuda", device_id=device_id
            )
        )


# io_binding.bind_ortvalue_output(
#     
# )

# print(inputs)
session.run_with_iobinding(io_binding)