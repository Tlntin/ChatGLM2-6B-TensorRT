import os
# from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import sys
import time
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

# save input tensor
pt_input_path1 = os.path.join(output_dir, "pt_input1.pt")
pt_input_path2 = os.path.join(output_dir, "pt_input2.pt")
pt_input_dict1 = dict()
pt_input_dict2 = dict()
# save output tensor
pt_output_path1 = os.path.join(output_dir, "pt_output1.pt")
pt_output_path2 = os.path.join(output_dir, "pt_output2.pt")
pt_output_dict1 = dict()
pt_output_dict2 = dict()


class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])

model_dir = os.path.join(project_dir, "chatglm_6b")
tokenizer = ChatGLMTokenizer.from_pretrained(model_dir)
model = ChatGLMForConditionalGeneration.from_pretrained(model_dir)
model = model.half().cuda()
model.eval()

input_text = "你好"
# test chat speed
all_res = []
st = time.time()
for i in range(10):
    responses, history = model.chat(tokenizer=tokenizer, query=input_text)
    all_res.append(responses)
et = time.time()
print(all_res)
tokens = tokenizer.encode("".join(all_res), return_tensors="pt")[0]
token_num = len(tokens)
speed = round(token_num / (et - st), 1)
print("speed: {} tokens/s".format(speed))
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
# save input1
pt_input_dict1["input_ids"] = input_ids.detach().cpu()
pt_input_dict1["position_ids"] = position_ids.detach().cpu()
pt_input_dict1["attention_mask"] = attention_mask.detach().cpu()
input_container1 = torch.jit.script(Container(pt_input_dict1))
input_container1.save(pt_input_path1)

output_dict1 = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
)

# save output1 logists
pt_output_dict1["logits"] = output_dict1["logits"].detach().cpu()
past_key_values_1 = output_dict1["past_key_values"]
print("one past_key_shape for input 1 is ", past_key_values_1[0][0].shape)
print("logits for input1 shape is ", output_dict1["logits"].shape)

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
print("one past_key_shape for input 2 is ", past_key_values_2[0][0].shape)
print("logits for input2 shape is ", output_dict2["logits"].shape)

# save input2
pt_input_dict2["input_ids"] = input_ids2.detach().cpu()
pt_input_dict2["position_ids"] = position_ids2.detach().cpu()
pt_input_dict2["attention_mask"] = attention_mask2.detach().cpu()

# save logits2
pt_output_dict2["logits"] = output_dict2["logits"].detach().cpu()

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
    present_key = past_key_values_1[layer_idx][0].detach().cpu()
    present_value = past_key_values_1[layer_idx][1].detach().cpu()
    pt_output_dict1[present_key_name] = present_key
    pt_output_dict1[present_value_name] = present_value

    # save input2 past_key_values
    # input2 past_key_values is same as output1 present_key_values
    pt_input_dict2[past_key_name] = present_key
    pt_input_dict2[past_value_name] = present_value

    # save output2 present_key_values
    present_key2 = past_key_values_2[layer_idx][0].detach().cpu()
    present_value2 = past_key_values_2[layer_idx][1].detach().cpu()
    pt_output_dict2[present_key_name] = present_key2
    pt_output_dict2[present_value_name] = present_value2
    dynamic_axes.update({
        f"past_key_values.{layer_idx}.decorder.key": {
            1: "batch_size", 0: "past_seq_length"
        },
        f"past_key_values.{layer_idx}.decorder.value": {
            1: "batch_size", 0: "past_seq_length"
        },
    })

# save output1
output1_container = torch.jit.script(Container(pt_output_dict1))
output1_container.save(pt_output_path1)

# save input2
input2_container = torch.jit.script(Container(pt_input_dict2))
input2_container.save(pt_input_path2)

# save output2
output2_container = torch.jit.script(Container(pt_output_dict2))
output2_container.save(pt_output_path2)

past_key_values = [
    [torch.zeros(0, 1, 32, 128, device=device).half() for _ in range(2)]
    for _ in range(28)
]

# ---prepare for onnx export ---
query = "你好"
input_ids = tokenizer.encode(query, return_tensors="pt").cuda()
input_ids = torch.cat((input_ids, input_ids, input_ids), dim=0)
batch_size, seq_length = input_ids.shape
context_lengths = [seq.tolist().index(tokenizer.bos_token_id) for seq in input_ids]
attention_mask = torch.ones((batch_size, seq_length, seq_length), device=input_ids.device)
attention_mask.tril_()
for i, context_length in enumerate(context_lengths):
    attention_mask[i, :, :context_length] = 1
attention_mask.unsqueeze_(1)
attention_mask = (attention_mask < 0.5).bool()
MASK, gMASK = tokenizer.mask_token_id, tokenizer.gmask_token_id
is_gmasks = (input_ids == gMASK).to(torch.int32)
is_masks = (input_ids == MASK).to(torch.int32)
use_gmasks = torch.sum(is_gmasks, dim=1) > 0
mask_positions = torch.where(use_gmasks, torch.argmax(is_gmasks, dim=1), torch.argmax(is_masks, dim=1)).to(torch.int32).unsqueeze(1)
batch_size, seq_length = input_ids.shape
if use_gmasks is None:
    use_gmasks = [False] * batch_size
context_lengths = [seq.tolist().index(tokenizer.bos_token_id) for seq in input_ids]
position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
for i, context_length in enumerate(context_lengths):
    position_ids[i, context_length:] = mask_positions[i]
block_position_ids = [torch.cat((
    torch.zeros(context_length, dtype=torch.long, device=input_ids.device),
    torch.arange(seq_length - context_length, dtype=torch.long, device=input_ids.device) + 1
)) for context_length in context_lengths]
block_position_ids = torch.stack(block_position_ids, dim=0)
position_ids = torch.stack((position_ids, block_position_ids), dim=1)
past_key_values = tuple(tuple(torch.zeros(0, input_ids.size(0), 32, 128, device=input_ids.device).half() for _ in range(2)) for _ in range(28))
# to support onnx runtime
# input_ids = input_ids.to(torch.int32)
# position_ids = position_ids.to(torch.int32)
attention_mask = attention_mask.to(torch.bool)
outputs = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
)
next_past_key_values = outputs.past_key_values
fake_attention_mask = torch.cat((attention_mask,attention_mask), dim=3)
next_outputs = model.forward(
    input_ids=input_ids,
    position_ids=position_ids,
    attention_mask=fake_attention_mask,
    past_key_values=next_past_key_values,
)
print("input_ids shape:", input_ids.shape, "; type:", input_ids.dtype)
print("position_ids shape:", position_ids.shape, "; type: ", input_ids.dtype)
print("fake attention_mask shape:",fake_attention_mask.shape, "; type: ", fake_attention_mask.dtype)
print("on past_key_value shape: ", next_past_key_values[0][0].shape, "; type:", next_past_key_values[0][0].dtype)

with torch.no_grad():
    torch.onnx.export(
        model,
        (input_ids,position_ids,fake_attention_mask,next_past_key_values),
        f=onnx_model_path,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        training=torch.onnx.TrainingMode.EVAL,
    )