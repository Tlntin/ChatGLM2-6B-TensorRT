import os
import sys
import os.path
import torch
import tensorrt as trt
from colored import stylize, fg
from typing import List, Tuple
from cuda import cudart


class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity == trt.ILogger.ERROR:
            print(stylize("[ERROR] " + msg, fg("red")))  # 红色字体
        elif severity == trt.ILogger.WARNING:
            print(stylize("[WARNING] " + msg, fg("yellow")))  # 黄色字体
        elif severity == trt.ILogger.INTERNAL_ERROR:
            print(stylize("[INTERNAL_ERROR] " + msg, fg("red")))  # 红色字体
        elif severity == trt.ILogger.INFO:
            print(stylize("[INFO] " + msg, fg("green")))  # 绿色字体
        elif severity == trt.ILogger.VERBOSE:
            print(stylize("[VERBOSE] " + msg, fg("blue")))  # 蓝色字体
        else:
            print("[UNKNOWN] " + msg)


def gpu_check(error: cudart.cudaError_t):
    if error != cudart.cudaError_t.cudaSuccess:
        error_name = cudart.cudaGetErrorName(error)
        error_info = cudart.cudaGetErrorString(error)
        print(stylize(f"ERROR [{error_name}]: {error_info}", fg("red")))
        raise Exception(f"ERROR [{error_name}]: {error_info}")


class KernelNoPast:
    def __init__(self, engine_path: str, batch_size: int = 1, num_layers: int = 28):
        assert os.path.exists(engine_path), print(f"{engine_path} not exists.")
        self.batch_size_ = batch_size
        self.n_input_ = 2 + num_layers * 2
        self.n_output_ = num_layers * 2 + 1
        self.n_total_ = self.n_input_ + self.n_output_
        self.tensor_names_ = []
        # self.logger_ = trt.Logger(trt.Logger.INFO)
        self.logger_ = MyLogger()
        self.runtime_ = trt.Runtime(self.logger_)
        # load engine
        with open(engine_path, "rb") as f:
            self.engine_ = self.runtime_.deserialize_cuda_engine(f.read())
        # verify io number
        self.verify_io_number()
        # init stream and context
        error_log, self.stream_ = cudart.cudaStreamCreate()
        gpu_check(error_log)
        self.context_ = self.engine_.create_execution_context()
        print(stylize("init context and stream OK", fg("green")))
        self.context_.set_optimization_profile_async(0, self.stream_)

    def __del__(self):
        cudart.cudaStreamDestroy(self.stream_)

    def verify_io_number(self):
        n_io = self.engine_.num_io_tensors
        if n_io != self.n_total_:
            raise RuntimeError(stylize(
                "Number of IO tensors is not correct, " +
                f"must be {self.n_total_}, but you have {n_io} tensors",
                fg("red")))
        n_input = 0
        n_output = 0
        for i in range(n_io):
            name = self.engine_.get_tensor_name(i)
            self.tensor_names_.append(name)
            if self.engine_.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                n_input += 1
            else:
                n_output += 1
        if n_input != self.n_input_:
            raise RuntimeError(stylize(
                "Number of input tensors is not correct, " +
                f"must be {self.n_input_}, but you have {n_input} tensors",
                fg("red")))
        if n_output != self.n_output_:
            raise RuntimeError(stylize(
                "Number of output tensors is not correct, " +
                f"must be {self.n_output_}, but you have {n_output} tensors",
                fg("red")))
        n_profile = self.engine_.num_optimization_profiles
        if n_profile != 1:
            raise RuntimeError(stylize(
                "Number of optimization profiles is not correct, " +
                f"must be 1, but you have {n_profile} profiles",
                fg("red")))
        print(stylize(f"number of profile: {n_profile}", fg("green")))

    def set_input_shape(self, seq_len: int):
        self.context_.set_input_shape("input_ids", (self.batch_size_, seq_len))
        self.context_.set_input_shape("position_ids", (self.batch_size_, seq_len))
        for i in range(2, self.n_input_):
            self.context_.set_input_shape(
                self.tensor_names_[i],
                (0, self.batch_size_, 2, 128)
            )


    def get_tensor_size(self):
        shape_list = []
        data_type_list = []
        for i in range(self.n_total_):
            tensor_name = self.tensor_names_[i]
            shape = self.context_.get_tensor_shape(tensor_name)
            shape_list.append(shape)
            data_type = self.engine_.get_tensor_dtype(tensor_name)
            data_type_list.append(data_type)
        return shape_list, data_type_list

    @staticmethod 
    def get_data_type(raw_data_type: trt.DataType):
        if raw_data_type == trt.DataType.FLOAT:
            torch_type = torch.float32
        elif raw_data_type == trt.DataType.HALF:
            torch_type = torch.float16
        elif raw_data_type == trt.DataType.INT32:
            torch_type = torch.int32
        elif raw_data_type == trt.DataType.INT8:
            torch_type = torch.int8
        else:
            raise Exception(f"not support type {raw_data_type}")
        return torch_type

    def forward(self, input_tensors: Tuple[torch.Tensor, torch.Tensor]):
        seq_len = input_tensors[0].size(1)
        device = input_tensors[0].device
        self.set_input_shape(seq_len)
        shape_list, data_type_list = self.get_tensor_size()
        # --- prepare for output --- #
        output_tensors = []
        for i in range(self.n_input_, self.n_total_):
            torch_type = self.get_data_type(data_type_list[i]) 
            tensor = torch.zeros(
                size=tuple(shape_list[i]), dtype=torch_type, device=device
            )
            output_tensors.append(tensor)
        # === run inference with V3 ===
        for i in range(2):
            self.context_.set_tensor_address(
                self.tensor_names_[i],
                input_tensors[i].data_ptr()
            )
        torch_type = self.get_data_type(data_type_list[2])
        sample_tensor = torch.zeros([1], dtype=torch_type, device=device)
        for i in range(2, self.n_input_):
            self.context_.set_tensor_address(
                self.tensor_names_[i],
                sample_tensor.data_ptr()
            )
        for i in range(self.n_input_, self.n_total_):
            self.context_.set_tensor_address(
                self.tensor_names_[i],
                output_tensors[i - self.n_input_].data_ptr()
            )
        self.context_.execute_async_v3(stream_handle=self.stream_)
        (error_info,) = cudart.cudaDeviceSynchronize()
        gpu_check(error_info)
        (error_info,) = cudart.cudaStreamSynchronize(self.stream_)
        gpu_check(error_info)
        return output_tensors


if __name__ == "__main__":
    now_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(now_dir)
    model_dir = os.path.join(project_dir, "models")
    engine_path1 = os.path.join(model_dir, "chatglm6b2-bs1_with_cache.plan")
    kernel = KernelNoPast(engine_path1)
    input_ids = torch.ones([1, 4], dtype=torch.int64).cuda()
    position_ids = torch.ones([1, 4], dtype=torch.int64).cuda()
    input_tensors1 = (input_ids, position_ids)
    output_tensors1 = kernel.forward(input_tensors1)
    print("first shape", output_tensors1[0].shape)
    print("last shape", output_tensors1[-1].shape)

