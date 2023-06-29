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

query = "想要出国留学，应该怎么办？"
history = [
    (
        "你好",
        "你好👋!我是人工智能助手 ChatGLM2-6B,很高兴见到你,欢迎问我任何问题。",
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

# --debug for chat --
# response, history = model.chat(tokenizer, query, history)
# print("res", response)
print("=" * 50)
print(" ---forward first --- ")
outputs = model.forward(
    **input_tensors
)

print(" ---forward first with no attention_mask --- ")
outputs2 = model.forward(
    input_ids=input_tensors["input_ids"],
    position_ids=input_tensors["position_ids"],
    attention_mask=torch.tensor([], device=device)
)


def compare_diff(outputs_1, outputs_2):
    print("--- compare diff ---")
    max_diff = 0
    logits_diff = (outputs_2["logits"] - outputs_1["logits"]).max().item()
    if logits_diff > max_diff:
        max_diff = logits_diff
    print("logits diff is ", logits_diff)
    past_key_values0 = outputs_1["past_key_values"]
    past_key_values1 = outputs_2["past_key_values"]
    for i in range(model.config.num_layers):
        present_key_name = f"present_key_values.{i}.key"
        present_value_name = f"present_key_values.{i}.value"
        diff1 = (past_key_values0[i][0] - past_key_values1[i][0]).max().item()
        diff2 = (past_key_values0[i][1] - past_key_values1[i][1]).max().item()
        print(f"{present_key_name} diff: ", diff1)
        print(f"{present_value_name} diff: ", diff2)
        if diff1 > max_diff:
            max_diff = diff1
        if diff2 > max_diff:
            max_diff = diff2

    print("max diff is: ", max_diff)


compare_diff(outputs, outputs2)
print("=" * 50)


print("=" * 50)
print(" ---forward second --- ")
attention_mask = input_tensors["attention_mask"]
position_ids = input_tensors["position_ids"]
past_key_values = outputs["past_key_values"]
# copy from forward in second time
input_ids = torch.tensor([[30910]]).to(device)
attention_mask = torch.cat(
    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
)
new_position_id = position_ids[..., -1:].clone()
new_position_id += 1
position_ids = torch.cat(
    [position_ids, new_position_id], dim=-1
)
# copy from prepare_inputs_for_generation in modeling_chatglm.py
position_ids = position_ids[..., -1:]
past_key_values1 = outputs["past_key_values"]
outputs_3 = model.forward(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values1
)

print(" ---forward second with no attention_mask --- ")
outputs_4 = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=torch.tensor([], device=device),
    past_key_values=past_key_values1
)
compare_diff(outputs_3, outputs_4)
print("=" * 50)


