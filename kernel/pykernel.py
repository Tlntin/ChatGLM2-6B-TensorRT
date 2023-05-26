import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torch.nn as nn

class Kernel(nn.Module):
    def __init__(self, engine_path: str, batch_size: int):
        self.logger_ = trt.Logger(trt.Logger.INFO)
        self.runtime_ = trt.Runtime(self.logger_)
        self.engine_ = None
        self.context_1_ = None
        self.context_2_ = None
        self.stream_1_ = cuda.Stream()
        self.stream_2_ = cuda.Stream()
        self.batch_size_ = batch_size
        self.n_total_ = 116
        self.n_input_ = 59
        self.n_output_ = 57
        self.tensor_names_ = []
        self.load_engine(engine_path)

    def __del__(self):
        pass

    def load_engine(self, engine_path: str):
        pass 

    def vertify_io_number(self):
        pass

    def init_execute_context(self):
        pass

    def forward(self, input_ids, position_ids, attention_mask, past_key_values=None):
        if past_key_values is None:
            context = self.context_1_
            seq_length = input_ids.shape[1]
            self.set_input_for_context1(seq_length)
        else:
            context = self.context_2_
            past_seq_length = past_key_values[0][0].shape[1]
            self.set_input_for_context2(past_seq_length)

    def set_input_for_context1(self, seq_length):
        pass

    def set_input_for_context2(self, past_seq_length):
        pass

    def run_gpu_inference(
            self,
            input_ids, 
            position_ids,
            attention_mask,
            past_key_values,
            bytes_list,
            type_bytes_list,
            context,
            stream
        ):
        pass




if __name__ == "__main__":
    kernel = Kernel("../models/chatglm6b-bs1-12.5G.plan", 1)