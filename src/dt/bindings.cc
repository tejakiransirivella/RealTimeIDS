#include "../../include/decision_tree.h"
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<pybind11/stl.h>

namespace py = pybind11;

/**
 * @brief Bindings for DecisionTree class
 */
PYBIND11_MODULE(decision_tree,m){
    py::class_<DecisionTree>(m, "DecisionTree")
    .def(py::init<>())
    .def("train", &DecisionTree::train)
    .def("predict", &DecisionTree::predict)
    .def("test", &DecisionTree::test)
    .def("loadData", &DecisionTree::loadData)
    .def("saveModel", &DecisionTree::saveModel)
    .def("loadModel", &DecisionTree::loadModel);
}