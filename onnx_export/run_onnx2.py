import onnxruntime as ort
import os
import numpy as np

os.environ["export ORT_LOG_LEVEL"] = "VERBOSE"


onnx_path = "../onnx_output/chatglm_6b.onnx"
sess = ort.InferenceSession(onnx_path)
input_ids = np.array([[19316]], dtype=np.int64)
position_ids = np.array([[[2], [2]]], dtype=np.int64)
attention_mask = np.array([[[[True]]]], dtype=np.bool_)

inputs = {
    "input_ids": input_ids,
    "position_ids": position_ids,
    "attention_mask": attention_mask,
}

for layer_idx in range(28):
    input_names = [
        f"past_key_values.{layer_idx}.decorder.key",
        f"past_key_values.{layer_idx}.decorder.value"
    ]
    past_key_values = np.zeros([4, 1, 32, 128], dtype=np.float32)
    inputs[input_names[0]] = past_key_values
    inputs[input_names[1]] = past_key_values

    # inputs[input_names[0]] = None
    # inputs[input_names[1]] = None
# print(inputs)
output = sess.run(
    ["logits"],
    input_feed=inputs    
  )

print(output)
print(len(output))
print(output[0].shape)