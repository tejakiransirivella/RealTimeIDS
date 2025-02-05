// #include "nn.h"
// #include<pybind11/pybind11.h>
// #include<pybind11/numpy.h>
// #include<pybind11/stl.h>

// namespace py = pybind11;

// // Neural network bindings
// PYBIND11_MODULE(ml,m){
//   py::class_<NeuralNetwork>(m,"NeuralNetwork")
//     .def(py::init<int,std::vector<int>,int,ActivationFunction>())
//     .def("setLossFunction",&NeuralNetwork::setLossFunction)
//     .def("predict",&NeuralNetwork::predict)
//     .def("print",&NeuralNetwork::print)
//     .def("train",&NeuralNetwork::train);
// }

// // Activation function bindings
// PYBIND11_MODULE(ml,m){
//   py::enum_<ActivationFunction>(m,"ActivationFunction")
//     .value("SIGMOID",SIGMOID)
//     .value("RELU",RELU);
// }