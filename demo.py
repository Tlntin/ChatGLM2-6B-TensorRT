import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import re
import time

from kernel import ckernel

Kernel = ckernel.Kernel
# from ckernel import Kernel
from kernel.logits_processor import *
# from kernel.ckernel import Kernel


class Model(nn.Module):
    def __init__(self, engine_path: str, batch_size: int):
        self.batch_size_ = batch_size
        self.kernel = Kernel(engine_path, batch_size)
        self.num_layers_ = 28
        self.logits_processor = None
        self.logits_warper = None

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple[torch.FloatTensor]] = None):
        if past_key_values is None:
            outputs = self.inference_step_1(input_ids, attention_mask, position_ids)
            return outputs
        else:
            outputs = self.inference_step_x(input_ids, input_ids, attention_mask, position_ids, past_key_values)
            return outputs

    def chat(
            self,
            tokenizer,
            query: str,
            history: List[Tuple[str, str]] = None,
            max_length: int = 2048,
            max_new_tokens: int = 40,
            num_beams=1,
            do_sample=True,
            top_p=0.7,
            top_k=50,
            temperature=1.0,
            **kwargs
    ):
        # 初始化 history
        if history is None:
            history = []
        # 初始化后处理
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(InvalidScoreLogitsProcessor())
        self.logits_warper = LogitsProcessorList()
        self.logits_warper.append(TemperatureLogitsWarper(temperature))
        self.logits_warper.append(TopPLogitsWarper(top_p))
        self.logits_warper.append(TopKLogitsWarper(top_k))
        # 组装prompt
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        # 第一次推理
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda().to(torch.int32)
        ori_len = len(input_ids[0])
        attention_mask, position_ids = self.pre_processing_step_1(tokenizer, input_ids)
        outputs_1 = self.kernel.forward(input_ids, position_ids, attention_mask)
        ori_input_ids, input_ids, attention_mask, position_ids, past_key_values = self.post_processing_step_1(
            outputs_1, input_ids, attention_mask, position_ids)
        # 重复推理直到条件终止
        while len(ori_input_ids[0]) < max_length \
                and len(ori_input_ids[0]) - ori_len < max_new_tokens \
                and tokenizer.eos_token_id not in ori_input_ids[0]:
            outputs_x = self.kernel.forward(input_ids, position_ids, attention_mask, past_key_values)
            ori_input_ids, input_ids, attention_mask, position_ids, past_key_values = self.post_processing_step_x(
                outputs_x, ori_input_ids, input_ids, attention_mask, position_ids)
            # print(tokenizer.decode(ori_input_ids[0]))
        # 处理回答
        response = tokenizer.decode(ori_input_ids[0][ori_len:])
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    def pre_processing_step_1(self, tokenizer, input_ids: torch.Tensor):
        BOS = tokenizer.bos_token_id
        MASK = tokenizer.mask_token_id
        gMASK = tokenizer.gmask_token_id
        batch_size, seq_length = input_ids.shape
        # 输入张量扩展常量
        input_range = torch.arange(seq_length, dtype=torch.int32).repeat((batch_size, 1)).to(input_ids.device)
        input_upper = torch.tril(torch.ones((batch_size, seq_length, seq_length), dtype=torch.int32)).to(
            input_ids.device)
        # 获取 attention_mask
        context_lengths = torch.argmax((input_ids == BOS).to(torch.int32), dim=1)
        context_mask = (input_range + 1) <= context_lengths.unsqueeze(1)
        padding_mask = context_mask.unsqueeze(1)
        attention_mask = torch.logical_not(torch.logical_or(input_upper, padding_mask)).unsqueeze(1)
        # 判断MASK位置
        is_gmasks = (input_ids == gMASK).to(torch.int32)
        is_masks = (input_ids == MASK).to(torch.int32)
        use_gmasks = torch.sum(is_gmasks, dim=1) > 0
        # 获取 position_ids
        mask_positions = torch.where(use_gmasks, torch.argmax(is_gmasks, dim=1), torch.argmax(is_masks, dim=1)).to(
            torch.int32).unsqueeze(1)
        position_ids_pre = torch.where(context_mask, input_range, mask_positions)
        block_position_ids = torch.clamp(input_range - context_lengths.unsqueeze(1) + 1, min=0)
        position_ids = torch.stack((position_ids_pre, block_position_ids), dim=1).to(torch.int32)
        return attention_mask, position_ids

    def post_processing_step_1(self,
                               outputs_1,
                               input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               position_ids: torch.Tensor):
        logits = outputs_1[-1]
        next_token_logits = logits[:, -1, :]
        # 一些后处理逻辑
        next_token_scores = self.logits_processor(input_ids, next_token_logits)
        next_token_scores = self.logits_warper(input_ids, next_token_scores)
        # 采样下一个token
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        ori_input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1)
        # 输出下一轮的 input_ids, position_ids, attention_mask
        attention_mask = attention_mask[..., -1:, -1:]
        position_ids = torch.cat(
            (position_ids[:, :-1, -1:], position_ids[:, -1:, -1:] + torch.tensor(1, dtype=position_ids.dtype)), dim=1)
        input_ids = ori_input_ids[:, -1:].to(torch.int32)
        past_key_values = ()
        for i in range(self.num_layers_):
            past_key = outputs_1[2 * i]
            past_value = outputs_1[2 * i + 1]
            past_key_values += ((past_key, past_value),)
        return ori_input_ids, input_ids, attention_mask, position_ids, past_key_values

    def post_processing_step_x(self,
                               outputs_x,
                               ori_input_ids: torch.Tensor,
                               input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               position_ids: torch.Tensor):
        logits = outputs_x[-1]
        next_token_logits = logits[:, -1, :]
        # 一些后处理逻辑
        next_token_scores = self.logits_processor(input_ids, next_token_logits)
        next_token_scores = self.logits_warper(input_ids, next_token_scores)
        # 采样下一个token
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        ori_input_ids = torch.cat((ori_input_ids, next_tokens[:, None]), dim=-1)
        # 输出下一轮的 input_ids, position_ids, attention_mask
        attention_mask = attention_mask[..., -1:, -1:]
        position_ids = torch.cat(
            (position_ids[:, :-1, -1:], position_ids[:, -1:, -1:] + torch.tensor(1, dtype=position_ids.dtype)), dim=1)
        input_ids = ori_input_ids[:, -1:].to(torch.int32)
        past_key_values = ()
        for i in range(self.num_layers_):
            past_key = outputs_x[2 * i]
            past_value = outputs_x[2 * i + 1]
            past_key_values += ((past_key, past_value),)
        return ori_input_ids, input_ids, attention_mask, position_ids, past_key_values

    def process_response(self, response):
        response = response.strip()
        # response = response.replace("[[训练时间]]", "2023年")
        punkts = [
            [",", "，"],
            ["!", "！"],
            [":", "："],
            [";", "；"],
            ["\?", "？"],
        ]
        for item in punkts:
            response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
            response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
        return response


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import time
    tokenizer = AutoTokenizer.from_pretrained("chatglm_6b", trust_remote_code=True)
    model = Model("models/chatglm6b-bs1-12.5G.plan", 1)
    all_res = []
    st = time.time()
    for i in range(10):
        responses, history = model.chat(tokenizer=tokenizer, query="你好")
        all_res.append(responses)
    et = time.time()
    print(all_res)
    tokens = tokenizer.encode("".join(all_res), return_tensors="pt")[0]
    token_num = len(tokens)
    speed = round(token_num / (et - st), 1)
    print("speed: {} tokens/s".format(speed))
    
