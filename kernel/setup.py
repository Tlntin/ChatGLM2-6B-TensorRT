import os
import torch
import pybind11
from setuptools import setup, Extension
from torch.utils import cpp_extension

input_dirs = cpp_extension.include_paths()
now_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(now_dir)
custom_include_dir = os.path.join(project_dir, "include")
cuda_include_dir = "/usr/local/cuda/include"
input_dirs += [custom_include_dir, cuda_include_dir]
library_dirs = ["/usr/local/cuda/lib64"]
libraries = ["nvinfer"]
torch_lib_dir = os.path.join(torch.__path__[0], "lib")
library_dirs.append(torch_lib_dir)
libraries.extend(["torch", "torch_cuda", "torch_cpu", "c10", "cudart", "c10_cuda"])
library_dirs.append(pybind11.get_include(False))
# extra_link_args = ["-D_GLIBCXX_USE_CXX11_ABI=1", "-std=c++17"]
extra_link_args = ["-D_GLIBCXX_USE_CXX11_ABI=0"]
#   "-L/usr/local/cuda/lib64", "-lnvinfer", f"-L{torch_lib_dir}"]

setup(name='ckernel',
      version="0.0.1",
      ext_modules=[
        cpp_extension.CppExtension(
          name='ckernel',
          sources=['kernel.cpp', "bind.cpp"],
          include_dirs=input_dirs,
          language="c++",
          extra_link_args=extra_link_args,
          libraries=libraries,
          library_dirs=library_dirs
        )
      ],
      cmdclass={'kernel': cpp_extension.BuildExtension}
    )