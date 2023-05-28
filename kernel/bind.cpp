#include <string>
#include <kernel.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(example, m) {
    py::class_<Kernel, std::shared_ptr<Kernel>>(m, "Kernel")
        .def(py::init<const std::string&, int>())
        .def("forward", (std::vector<torch::Tensor> (Kernel::*)(
            const torch::Tensor&,
            const torch::Tensor&,
            const torch::Tensor&)
        ) &Kernel::forward)
        .def("forward", (std::vector<torch::Tensor> (Kernel::*)(
            const torch::Tensor&,
            const torch::Tensor&,
            const torch::Tensor&,
            const std::vector<std::vector<torch::Tensor>>&)
        ) &Kernel::forward);
}
