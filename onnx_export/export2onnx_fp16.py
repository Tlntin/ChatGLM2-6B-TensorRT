import os
# from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import numpy as np
import sys
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)
from chatglm_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm_6b.tokenization_chatglm import ChatGLMTokenizer


output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
onnx_output_dir = os.path.join(output_dir, "onnx_output")
if not os.path.exists(onnx_output_dir):
    os.mkdir(onnx_output_dir)
else:
    for file in os.listdir(onnx_output_dir):
        os.remove(os.path.join(onnx_output_dir, file))
onnx_model_path = os.path.join(onnx_output_dir, "chatglm_6b.onnx")


model_dir = os.path.join(project_dir, "chatglm_6b")
tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
model = ChatGLMForConditionalGeneration.from_pretrained(model_dir)
model = model.half().cuda()
model.eval()

input_text = "你好"
response, history = model.chat(tokenizer, input_text, history=[])
print(response)
# model = model.cpu().float()
# model.eval()

device = 'cuda'
input_ids = tokenizer([input_text], return_tensors="pt")["input_ids"]
input_ids = input_ids.to(device=device).int()
position_ids = torch.tensor([[[0, 1, 2, 2], [0, 0, 0, 1]]], device=device).int()
attention_mask = torch.tensor(
    [[[
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, False]
    ]]], device=device, dtype=torch.bool
)

output_dict = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
)

# save input1
np_input_dir1 = os.path.join(output_dir, "np_input1")
if not os.path.exists(np_input_dir1):
    os.mkdir(np_input_dir1)
input_ids_path = os.path.join(np_input_dir1, "input_ids.npy")
position_ids_path = os.path.join(np_input_dir1, "position_ids.npy")
attention_mask_path = os.path.join(np_input_dir1, "attention_mask.npy")
input_ids_np = input_ids.cpu().numpy()
position_ids_np = position_ids.cpu().numpy()
attention_mask_np = attention_mask.cpu().numpy()
np.save(input_ids_path, input_ids_np)
np.save(position_ids_path, position_ids_np)
np.save(attention_mask_path, attention_mask_np)

# save output1 logists
np_output_dir1 = os.path.join(output_dir, "np_output1")
if not os.path.exists(np_output_dir1):
    os.mkdir(np_output_dir1)
logits_path = os.path.join(np_output_dir1, "logits.npy")
logits = output_dict["logits"].cpu().float().detach().data.numpy()
np.save(logits_path, logits)

_past_key_values = output_dict["past_key_values"]
print("one past_key_shape", _past_key_values[0][0].shape)
past_key_values = [
    [torch.zeros(0, 1, 32, 128, device=device).half() for _ in range(2)]
    for _ in range(28)
]

input_names=["input_ids",'position_ids', "attention_mask"]
output_names=["logits"]
dynamic_axes={
    'input_ids': {0: "batch_size", 1: "seq_length"},
    'position_ids':{0: "batch_size", 2: "seq_length" },
    "attention_mask":{0: "batch_size", 2: "seq_length", 3:"seq_length" }
}

for layer_idx in range(model.config.num_layers):
    input_names += [
        f"past_key_values.{layer_idx}.decorder.key",
        f"past_key_values.{layer_idx}.decorder.value"
    ]
    output_names += [
        f"present_key_values.{layer_idx}.decorder.key",
        f"present_key_values.{layer_idx}.decorder.value"
    ]
    # save output1 present_key_values 
    present_key_path = os.path.join(
        np_output_dir1, f"present_key_values.{layer_idx}.decorder.key.npy"
    )
    present_value_path = os.path.join(
        np_output_dir1, f"present_key_values.{layer_idx}.decorder.value.npy"
    )
    present_key = _past_key_values[layer_idx][0].cpu().float().detach().data.numpy()
    present_value = _past_key_values[layer_idx][1].cpu().float().detach().data.numpy()
    np.save(present_key_path, present_key)
    np.save(present_value_path, present_value)

    dynamic_axes.update({
        f"past_key_values.{layer_idx}.decorder.key": {
            1: "batch_size", 0: "past_seq_length"
        },
        f"past_key_values.{layer_idx}.decorder.value": {
            1: "batch_size", 0: "past_seq_length"
        },
    })

"""
with torch.no_grad():
    torch.onnx.export(
        model, (input_ids,position_ids,attention_mask, past_key_values),
        "../onnx_output/chatglm_6b.onnx",
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )
"""