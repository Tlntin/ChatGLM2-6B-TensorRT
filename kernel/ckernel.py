import os
import torch
import pybind11
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load

input_dirs = cpp_extension.include_paths()
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
custom_include_dir = os.path.join(project_dir, "include")
cuda_include_dir = "/usr/local/cuda/include"
input_dirs += [custom_include_dir, cuda_include_dir]

torch_dir = os.path.dirname(torch.__path__[0])
torch_lib_dir = os.path.join(torch_dir, "lib")
library_dirs = ["/usr/local/cuda/lib64", torch_lib_dir]
libraries = ["nvinfer"]
torch_lib_dir = os.path.join(torch.__path__[0], "lib")
library_dirs.append(torch_lib_dir)
libraries.extend(["torch", "torch_cuda", "torch_cpu", "c10", "cudart", "c10_cuda"])
library_dirs.append(pybind11.get_include(False))
extra_link_args = [
    "-std=c++17",
    "-L/usr/local/cuda/lib64",
    f"-L{torch_lib_dir}",
  ]
extra_cuda_cflags = [
    "-std=c++17",
    "-L/usr/local/cuda/lib64",
    f"-L{torch_lib_dir}",
    "-lnvinfer"
]


sources=['kernel.cpp', "bind.cpp"]
sources = [os.path.join(now_dir, s) for s in sources]
ckernel = load(
  name='ckernel',
  sources=sources,
  extra_include_paths=input_dirs,
  extra_cflags=extra_link_args,
  extra_cuda_cflags=extra_cuda_cflags,
  extra_ldflags=extra_cuda_cflags,
  with_cuda=True,
  verbose=False
)
# print(ckernel)
# print(ckernel.Kernel)
# print(ckernel.Kernel.forward)
