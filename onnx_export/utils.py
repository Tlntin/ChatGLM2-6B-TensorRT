import torch


def get_input_tensors(query, tokenizer, device):
    input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    # input_ids = torch.cat((input_ids, input_ids), dim=0)
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
    return input_ids, position_ids, attention_mask


def get_prompt(query, history=None):
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    return prompt