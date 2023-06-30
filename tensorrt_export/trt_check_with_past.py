import os
import sys
import torch
from colored import stylize, fg


now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
sys.path.append(project_dir)


# from kernel.pykernel_no_past import KernelNoPast
from kernel.pykernel_with_past import KernelWithPast



def check_value(pre_value: torch.Tensor, true_value: torch.Tensor, diff=1e-3):
    if pre_value.shape != true_value.shape:
        raise Exception("compare shape must be same!")
    max_diff = (pre_value - true_value).abs_().max().item()
    if max_diff > diff:
        print(stylize(f"compare diff failed, diff is {max_diff}", fg("red")))
    else:
        print(stylize("compare diff OK!", fg("green")))
    return max_diff


def main():
    assert torch.cuda.is_available(), print("you must has cuda to run TensorRT")
    output_dir = os.path.join(project_dir, "output")
    model_dir = os.path.join(project_dir, "models")
    engine_path1 = os.path.join(model_dir, "chatglm6b2-bs1_with_cache.plan")
    input_path = os.path.join(output_dir, "pt_input2.pt")
    output_path = os.path.join(output_dir, "pt_output2.pt")
    device = torch.device("cuda:0")
    input_dict = torch.jit.load(input_path)
    batch_size = 1
    num_layers = 28
    output_dict = torch.jit.load(output_path)
    input_ids: torch.Tensor = input_dict.input_ids.int().to(device)
    position_ids: torch.Tensor = input_dict.position_ids.int().to(device)
    input_tensors = [input_ids, position_ids]
    for i in range(num_layers):
        input_names = [
            f"past_key_values.{i}.key",
            f"past_key_values.{i}.value"
        ]
        for name in input_names:
            one_key_value = getattr(input_dict, name).to(device)
            input_tensors.append(one_key_value)
    kernel = KernelWithPast(engine_path1, batch_size, num_layers)
    output_tensors = kernel.forward(tuple(input_tensors))
    # compare output
    max_diff_ = 0
    # compare logits
    logits = output_dict.logits.to(device)
    pred_logits = output_tensors[-1]
    logits_diff = check_value(logits, pred_logits)
    print("=" * 20)
    print("compare logits")
    if logits_diff > max_diff_:
        max_diff_ = logits_diff
    # compare past key values
    for i in range(num_layers):
        present_key_name = f"present_key_values.{i}.key"
        present_value_name = f"present_key_values.{i}.value"
        true_present_key = getattr(output_dict, present_key_name).to(device)
        true_present_value = getattr(output_dict, present_value_name).to(device)
        pre_present_key = output_tensors[i * 2]
        pre_present_value = output_tensors[i * 2 + 1]
        print("=" * 20)
        print("compare: ", present_key_name)
        temp_diff = check_value(pre_present_key, true_present_key)
        if temp_diff > max_diff_:
            max_diff_ = temp_diff

        print("=" * 20)
        print("compare: ", present_value_name)
        temp_diff = check_value(pre_present_value, true_present_value)
        if temp_diff > max_diff_:
            max_diff_ = temp_diff
    print(f"max diff is {max_diff_}")


if __name__ == "__main__":
    main()



