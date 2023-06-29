import os
# from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import sys
import argparse
from transformers.generation.utils import LogitsProcessorList
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)
from chatglm2_6b.configuration_chatglm import ChatGLMConfig
from chatglm2_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm2_6b.tokenization_chatglm import ChatGLMTokenizer
from onnx_export.utils import build_inputs
from transformers.models.bloom import BloomOnnxConfig


parser = argparse.ArgumentParser(description='export pytorch model to onnx')
parser.add_argument(
    '--data_type',
    default="fp32",
    help='use fp16/fp32 to export onnx model. if use fp16, you need GPU memory > 24G, defualt is fp32'
)

args = parser.parse_args()
if args.data_type == "fp16":
    device = 'cuda'
else:
    device = 'cpu'


output_dir = os.path.join(project_dir, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
onnx_output_dir = os.path.join(output_dir, "onnx_output")
if not os.path.exists(onnx_output_dir):
    os.mkdir(onnx_output_dir)
else:
    for file in os.listdir(onnx_output_dir):
        os.remove(os.path.join(onnx_output_dir, file))
onnx_model_path = os.path.join(onnx_output_dir, "chatglm2_6b.onnx")

query = "想要出国留学，应该怎么办？"
history = [
    (
        "你好",
        "你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。",
    )
]

model_dir = os.path.join(project_dir, "chatglm2_6b")
tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
config = ChatGLMConfig.from_pretrained(model_dir)
# config.num_layers = 1
model = ChatGLMForConditionalGeneration.from_pretrained(model_dir, config=config)
if device == "cuda":
    model = model.half().cuda()
else:
    model = model.float().cpu()
device = torch.device(device)
model.eval()
# input_tensors
input_tensors = build_inputs(device, tokenizer, query, history)
del input_tensors["attention_mask"]
# --debug for chat --
# response, history = model.chat(tokenizer, query, history)
# print("res", response)

print(" ---forward first --- ")
outputs = model.forward(
    **input_tensors
)

print("--second forward ---")
# input_ids = input_tensors["input_ids"]
position_ids = input_tensors["position_ids"]
past_key_values = outputs["past_key_values"]
# copy from forward in second time
new_input_ids = torch.tensor([[30910]]).to(device)
input_ids = torch.cat([input_tensors["input_ids"], new_input_ids], dim=-1)

# copy from _update_model_kwargs_for_generation in modeling_chatglm.py
new_position_id = position_ids[..., -1:].clone()
new_position_id += 1
position_ids = torch.cat(
    [position_ids, new_position_id], dim=-1
)
# copy from prepare_inputs_for_generation in modeling_chatglm.py
# position_ids = position_ids[..., -1:]
# input_ids = input_ids[..., -1:]
# print shape
print(
    "input_ids shape:", input_ids.shape,
    "; type:", input_ids.dtype
)
print(
    "position_ids shape:", position_ids.shape,
    "; type: ", input_ids.dtype
)
print(
    "first forward one past_key_value shape: ", past_key_values[0][0].shape,
    "; type:", past_key_values[0][0].dtype
)
print("first forward logits shape: ", outputs["logits"].shape)
outputs2 = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    past_key_values=past_key_values
)
print("--- export onnx ---")
# ---prepare for onnx export ---
input_names = ["input_ids", 'position_ids']
output_names = ["logits"]
dynamic_axes = {
    'input_ids': {0: "batch_size", 1: "sequence"},
    'position_ids': {0: "batch_size", 1: "sequence"},
    "logits": {0: "batch_size", 1: "sequence"}
}
for layer_idx in range(model.config.num_layers):
    # --- input key and value ---
    past_key_name = f"past_key_values.{layer_idx}.key"
    past_value_name = f"past_key_values.{layer_idx}.value"
    input_names += [past_key_name, past_value_name]
    # --- output key and value ---
    present_key_name = f"present_key_values.{layer_idx}.key"
    present_value_name = f"present_key_values.{layer_idx}.value"
    output_names += [present_key_name, present_value_name]
    dynamic_axes.update({
        past_key_name: {
            0: "past_sequence",
            1: "batch_size",
        },
        past_value_name: {
            0: "past_sequence",
            1: "batch_size",
        },
        present_key_name: {
            0: "past_sequence + sequence",
            1: "batch_size"
        },
        present_value_name: {
            0: "past_sequence + sequence",
            1: "batch_size"
        }
    })


with torch.no_grad():
    torch.onnx.export(
        model,
        args=(
            input_ids,
            position_ids,
            past_key_values
        ),
        f=onnx_model_path,
        opset_version=14,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )