import onnxruntime as ort
# from transformers import AutoTokenizer
import numpy as np
import os

os.environ["export ORT_LOG_LEVEL"] = "VERBOSE"


#  tokenizer = AutoTokenizer.from_pretrained("../chatglm_6b", trust_remote_code=True)

onnx_path = "../output/onnx_output/chatglm_6b.onnx"
providers = ["CPUExecutionProvider"]
sess = ort.InferenceSession(onnx_path, providers=providers)
input_text = "你好"
# input_ids = tokenizer([input_text], return_tensors="pt")["input_ids"].data.numpy()
# print(input_ids)
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
    past_key_values = np.zeros([0, 1, 32, 128], dtype=np.float32)
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