#include <string>
#include <kernel.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(kernel, m) {
    m.doc() = "pybind11 kernel with tensorRT compiled for ChatGLM-6b"; // optional module docstring
    py::class_<Kernel>(m, "Kernel")
        .def(py::init<const std::string &, int>(), py::arg("engine_path"), py::arg("batch_size"))
        .def("forward", (std::vector<std::vector<__half>> (Kernel::*)(std::vector<int>&, std::vector<int>&, std::vector<bool>&)) &Kernel::forward)
        .def("forward", (std::vector<std::vector<__half>> (Kernel::*)(std::vector<int>&, std::vector<int>&, std::vector<bool>&, std::vector<std::vector<std::vector<__half>>>&)) &Kernel::forward);
}
