#include "nn.h"
#include "tensor.h"
#include "decision_tree.h"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

namespace py = pybind11;

// Neural network bindings
PYBIND11_MODULE(ml,m){
      py::class_<NeuralNetwork>(m,"NeuralNetwork")
    .def(py::init<int,std::vector<int>,int,ActivationFunction>())
    .def("setLossFunction",&NeuralNetwork::setLossFunction)
    .def("setBatchSize",&NeuralNetwork::setBatchSize)
    .def("showLoss",&NeuralNetwork::showLoss)
    .def("predict",&NeuralNetwork::predict)
    .def("print",&NeuralNetwork::print)
    .def("train",&NeuralNetwork::train)
    .def("saveModel",&NeuralNetwork::saveModel)
    .def("loadModel",&NeuralNetwork::loadModel);

      py::enum_<ActivationFunction>(m,"ActivationFunction")
    .value("SIGMOID",SIGMOID)
    .value("RELU",RELU);

    py::enum_<LossFunction>(m,"LossFunction")
    .value("MSE",MSE)
    .value("CROSS_ENTROPY",CROSS_ENTROPY);

     py::class_<Tensor>(m,"Tensor")
    .def(py::init<std::vector<size_t>>())
    .def(py::init<std::vector<size_t>,std::vector<std::vector<size_t>>,std::vector<double>>())
    .def_static("ones",&Tensor::ones)
    .def("index",&Tensor::index)
    .def("reshape",&Tensor::reshape)
    .def("transpose",&Tensor::transpose)
    .def("neg",&Tensor::neg)
    .def("reciprocal",&Tensor::reciprocal)
    .def("add",&Tensor::add)
    .def("subtract",&Tensor::subtract)
    .def("mult",&Tensor::mult)
    .def("elementwise_mult",&Tensor::elementwise_mult)
    .def("pow",&Tensor::pow)
    .def("relu",&Tensor::relu)
    .def("binarilize",&Tensor::binarilize)
    .def("exp",&Tensor::exp)
    .def("matmul",&Tensor::matmul)
    .def("print",&Tensor::print)
    .def("get_data",&Tensor::get_data)
    .def("get_dims",&Tensor::get_dims)
    .def("get_gpu_data",&Tensor::get_gpu_data);

    py::class_<DecisionTree>(m, "DecisionTree")
    .def(py::init<>())
    .def("train", &DecisionTree::train)
    .def("predict", &DecisionTree::predict)
    .def("test", &DecisionTree::test)
    .def("loadData", &DecisionTree::loadData)
    .def("saveModel", &DecisionTree::saveModel)
    .def("loadModel", &DecisionTree::loadModel);
}