import os
# from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import sys
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)
from chatglm_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm_6b.tokenization_chatglm import ChatGLMTokenizer

model_dir = os.path.join(project_dir, "chatglm_6b")
tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
model = ChatGLMForConditionalGeneration.from_pretrained(model_dir).float().cpu()


input_text = "."
response, history = model.chat(tokenizer, input_text, history=[])
print(response)
model = model.cpu().float()
model.eval()

device = 'cpu'
input_ids = tokenizer([input_text], return_tensors="pt")["input_ids"]
input_ids = input_ids.to(device=device)
position_ids = torch.tensor([[[0, 1, 2, 2], [0, 0, 0, 1]]], device=device)
attention_mask = torch.tensor(
    [[[
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, True],
        [False, False, False, False]
    ]]], device=device, dtype=torch.bool
)
# output_dict = model.forward(
#     input_ids=input_ids,
#     position_ids=position_ids,
#     attention_mask=attention_mask,
# )
# past_key_values = output_dict["past_key_values"]
# print("one past_key_shape", past_key_values[0][0].shape)
past_key_values = [
    [torch.zeros(0, 1, 32, 128) for _ in range(2)]
    for _ in range(28)
]
if not os.path.exists("../onnx_output"):
    os.makedirs("../onnx_output")
else:
    for file in os.listdir("../onnx_output"):
        os.remove(os.path.join("../onnx_output", file))


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

    dynamic_axes.update({
        f"past_key_values.{layer_idx}.decorder.key": {
            1: "batch_size", 0: "past_seq_length"
        },
        f"past_key_values.{layer_idx}.decorder.value": {
            1: "batch_size", 0: "past_seq_length"
        },
    })
print("=======input_names=======")
print(input_names)
print("=======dynamic_axes=========")
print(dynamic_axes)

torch.onnx.export(
    model, (input_ids,position_ids,attention_mask, past_key_values),
    "../onnx_output/chatglm_6b.onnx",
    opset_version=18,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    training=torch.onnx.TrainingMode.EVAL,
)