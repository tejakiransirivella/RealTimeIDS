#include<vector> 
#include "tensor.h"
enum class Operation {
    RESHAPE,
    TRANSPOSE,
    NEG,
    RECIPROCAL,
    ADD,
    SUBTRACT,
    MULTIPLY,
    ELEMENTWISE_MULTIPLY,
    POW,
    RELU,
    BINARILIZE,
    EXP,
    MATMUL,
    PRINT,
};

class Node {
public:
    std::vector<Node> children;
    void add_child(Node child) {
        children.push_back(child);
    }
};

class SourceNode : public Node {
public:
    Tensor *tensor;
    SourceNode(Tensor *tensor) : tensor(tensor) {}
};

class OperationNode : public Node {
public:
    Operation operation;
    OperationNode(Operation operation) : operation(operation) {}
};


class ComputationGraph {
public:
    std::vector<SourceNode> nodes;

    void add_source_node(Tensor *tensor) {
        nodes.emplace_back(SourceNode(tensor));
    }

    void compute() {
        // Implement the logic to compute the graph
    }
    
    void print() {
        // Implement the logic to print the graph
    }


};