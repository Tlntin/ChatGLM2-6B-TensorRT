#include <kernel.hpp>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


PYBIND11_MODULE(ckernel, m) {
    py::class_<Kernel>(m, "Kernel")
        .def(py::init<const std::string&, int>())
        .def("forward", py::overload_cast<
                const torch::Tensor &,
                const torch::Tensor &,
                const torch::Tensor &
            >(&Kernel::forward))
        .def("forward", py::overload_cast<
                const torch::Tensor &,
                const torch::Tensor &,
                const torch::Tensor &,
                const std::vector<std::vector<torch::Tensor>>&
            >(&Kernel::forward));
}
