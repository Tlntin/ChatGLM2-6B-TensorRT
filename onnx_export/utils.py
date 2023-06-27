import torch
from typing import List, Tuple


def build_inputs(device, tokenizer, query: str,
                 history: List[Tuple[str, str]] = None):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query,
                                                            response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    inputs = tokenizer([prompt], return_tensors="pt")
    inputs = inputs.to(device)
    return inputs
