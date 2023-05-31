#include <kernel.hpp>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


PYBIND11_MODULE(ckernel, m) {
    py::class_<Kernel>(m, "Kernel")
        .def(py::init<const std::string&, int>(), py::arg("model_path"), py::arg("batch_size"))
        .def("forward", &Kernel::forward, py::arg("input_tensors"));
}
