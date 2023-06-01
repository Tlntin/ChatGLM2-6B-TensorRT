import os
# from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import sys
import argparse
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)
from onnx_export.utils import get_prompt, get_input_tensors
from chatglm_6b.modeling_chatglm import ChatGLMForConditionalGeneration
from chatglm_6b.tokenization_chatglm import ChatGLMTokenizer


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
onnx_model_path = os.path.join(onnx_output_dir, "chatglm_6b.onnx")

query = "想要出国留学，应该怎么办？"
history = [
    (
        "你好",
        "你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。",
    )
]

prompt = get_prompt(query, history)
model_dir = os.path.join(project_dir, "chatglm_6b")
tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
model = ChatGLMForConditionalGeneration.from_pretrained(model_dir)
if device == "cuda":
    model = model.half().cuda()
else:
    model = model.float().cpu()
model.eval()


# ---prepare for onnx export ---
input_names=["input_ids",'position_ids', "attention_mask"]
output_names=["logits"]
dynamic_axes={
    'input_ids': {0: "batch_size", 1: "seq_length"},
    'position_ids':{0: "batch_size", 2: "seq_length" },
    "attention_mask":{0: "batch_size", 2: "seq_length", 3:"seq_length" }
}
for layer_idx in range(model.config.num_layers):
    # --- input key and value ---
    past_key_name = f"past_key_values.{layer_idx}.decorder.key"
    past_value_name = f"past_key_values.{layer_idx}.decorder.value"
    input_names += [past_key_name, past_value_name]
    # --- output key and value ---
    present_key_name = f"present_key_values.{layer_idx}.decorder.key"
    present_value_name = f"present_key_values.{layer_idx}.decorder.value"
    output_names += [present_key_name, present_value_name]
    dynamic_axes.update({
        f"past_key_values.{layer_idx}.decorder.key": {
            1: "batch_size", 0: "past_seq_length"
        },
        f"past_key_values.{layer_idx}.decorder.value": {
            1: "batch_size", 0: "past_seq_length"
        },
    })

input_ids, position_ids, attention_mask = get_input_tensors(
    prompt, tokenizer, device)
past_key_values = tuple(
    tuple(
        torch.zeros(0, input_ids.size(0), 32, 128).to(device).half()
        if device == "cuda"
        else torch.zeros(0, input_ids.size(0), 32, 128).to(device).float()
        for _ in range(2)
    )
    for _ in range(28)
)
# to support onnx runtime, use int64 to export
# input_ids = input_ids.to(torch.int32)
# position_ids = position_ids.to(torch.int32)
attention_mask = attention_mask.to(torch.bool)
outputs = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
)
next_past_key_values = outputs["past_key_values"]
# use fake attention mask to export onnx
fake_attention_mask = torch.cat((attention_mask, attention_mask), dim=3)
next_outputs = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=fake_attention_mask,
    past_key_values=next_past_key_values,
)
print(
    "input_ids shape:", input_ids.shape,
    "; type:", input_ids.dtype
)
print(
    "position_ids shape:", position_ids.shape,
    "; type: ", input_ids.dtype
)
print(
    "fake attention_mask shape:",fake_attention_mask.shape,
    "; type: ", fake_attention_mask.dtype
)
print(
    "on past_key_value shape: ", next_past_key_values[0][0].shape,
    "; type:", next_past_key_values[0][0].dtype
)

with torch.no_grad():
    torch.onnx.export(
        model,
        args=(
            input_ids,
            position_ids,
            fake_attention_mask, 
            next_past_key_values
        ),
        f=onnx_model_path,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )