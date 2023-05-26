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
# --- prepare data for input1 ---
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

input_ids_np = input_ids.data.cpu().numpy()
position_ids_np = position_ids.data.cpu().numpy()
attention_mask_np = attention_mask.data.cpu().numpy()
input_ids_path = os.path.join(np_input_dir1, "input_ids.npy")
position_ids_path = os.path.join(np_input_dir1, "position_ids.npy")
attention_mask_path = os.path.join(np_input_dir1, "attention_mask.npy")
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

past_key_values_1 = output_dict["past_key_values"]
print("one past_key_shape", past_key_values_1[0][0].shape)

# --- prepare data for input2 ---
input_ids2 = torch.tensor([[5]], device=device).int()
position_ids2 = torch.tensor([[[2], [2]]], device=device).int()
attention_mask2 = torch.tensor([[[[False]]]], device=device, dtype=torch.bool)
output_dict2 = model.forward(
    input_ids=input_ids2,
    position_ids=position_ids2,
    attention_mask=attention_mask2,
    past_key_values=past_key_values_1,
)
past_key_values_2 = output_dict2["past_key_values"]
# save input2
np_input_dir2 = os.path.join(output_dir, "np_input2")
if not os.path.exists(np_input_dir2):
    os.mkdir(np_input_dir2)
np_output_dir2 = os.path.join(output_dir, "np_output2")
if not os.path.exists(np_output_dir2):
    os.mkdir(np_output_dir2)
input_ids_path2 = os.path.join(np_input_dir2, "input_ids.npy")
position_ids_path2 = os.path.join(np_input_dir2, "position_ids.npy")
attention_mask_path2 = os.path.join(np_input_dir2, "attention_mask.npy")
input_ids_np2 = input_ids2.cpu().numpy()
position_ids_np2 = position_ids2.cpu().numpy()
attention_mask_np2 = attention_mask2.cpu().numpy()
np.save(input_ids_path2, input_ids_np2)
np.save(position_ids_path2, position_ids_np2)
np.save(attention_mask_path2, attention_mask_np2)


# save logits2
logits_path2 = os.path.join(np_output_dir2, "logits.npy")
logits2 = output_dict2["logits"].cpu().float().detach().data.numpy()
np.save(logits_path2, logits2)

# prepare for onnx export
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
    # save output1 present_key_values 
    present_key_path = os.path.join(np_output_dir1, f"{present_key_name}.npy")
    present_value_path = os.path.join(np_output_dir1, f"{present_value_name}.npy")
    present_key = past_key_values_1[layer_idx][0].cpu().float().detach().data.numpy()
    present_value = past_key_values_1[layer_idx][1].cpu().float().detach().data.numpy()
    np.save(present_key_path, present_key)
    np.save(present_value_path, present_value)

    # save input2 past_key_values
    # input2 past_key_values is same as output1 present_key_values
    past_key_path2 = os.path.join(np_input_dir2, f"{past_key_name}.npy")
    past_value_path2 = os.path.join(np_input_dir2, f"{past_value_name}.npy")
    np.save(past_key_path2, present_key)
    np.save(past_value_path2, present_value)

    # save output2 present_key_values
    present_key_path2 = os.path.join(np_output_dir2, f"{present_key_name}.npy")
    present_value_path2 = os.path.join(np_output_dir2, f"{present_value_name}.npy")
    present_key2 = past_key_values_2[layer_idx][0].cpu().float().detach().data.numpy()
    present_value2 = past_key_values_2[layer_idx][1].cpu().float().detach().data.numpy()
    np.save(present_key_path2, present_key2)
    np.save(present_value_path2, present_value2)

    dynamic_axes.update({
        f"past_key_values.{layer_idx}.decorder.key": {
            1: "batch_size", 0: "past_seq_length"
        },
        f"past_key_values.{layer_idx}.decorder.value": {
            1: "batch_size", 0: "past_seq_length"
        },
    })

past_key_values = [
    [torch.zeros(0, 1, 32, 128, device=device).half() for _ in range(2)]
    for _ in range(28)
]
with torch.no_grad():
    torch.onnx.export(
        model, (input_ids,position_ids,attention_mask, past_key_values),
        onnx_model_path,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )