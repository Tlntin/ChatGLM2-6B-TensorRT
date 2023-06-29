import os
import torch
from polygraphy.json import save_json
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
output_dir = os.path.join(project_dir, "output")
input_path1 = os.path.join(output_dir, "pt_input1.pt")
output_path1 = os.path.join(output_dir, "pt_output1.pt")
input_dict = torch.jit.load(input_path1)
output_dict = torch.jit.load(output_path1)
input_ids = input_dict.input_ids.numpy()
position_ids = input_dict.position_ids.numpy()


def load_data():
    for _ in range(5):
        yield {"input_ids": input_ids, "position_ids": position_ids}


input_data = list(load_data())
save_json(input_data, "custom_inputs.json", description="custom inputs")