#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tensor.h"
using namespace std;

enum LossFunction{
    MSE,
    CROSS_ENTROPY
};

// Neural Network class
class NeuralNetwork {
public:
    vector<Tensor> weights;
    vector<Tensor> biases;
    vector<Tensor> activations,linear_activations;
    vector<ActivationFunction> activation_functions;
    LossFunction loss_function;
    int batch_size;
    int curr_processed;
    bool show_loss;
    double accumulated_loss;
    NeuralNetwork(int input_nodes,vector<int> hidden_nodes,int output_nodes,ActivationFunction activation):NeuralNetwork(input_nodes,hidden_nodes,output_nodes,vector<ActivationFunction>(hidden_nodes.size()+1,activation)) {

    }

    NeuralNetwork(int input_nodes,vector<int> hidden_nodes,int output_nodes,vector<ActivationFunction> activationFunctions) {
        srand(time(0));
        initializeWeights(input_nodes, hidden_nodes, output_nodes);
        initializeBiases(hidden_nodes, output_nodes);
        initializeActivations(input_nodes, hidden_nodes, output_nodes);
        activation_functions = activationFunctions;
        loss_function = MSE;
        batch_size = 1;
        curr_processed = 0;
        show_loss = false;
        accumulated_loss = 0;
    }

    void setLossFunction(LossFunction lossFunction) {
        loss_function = lossFunction;
    }

    void setBatchSize(int batchSize) {
        batch_size = batchSize;
    }

    void showLoss(bool show) {
        show_loss = show;
    }

    void initializeWeights(int input_nodes,vector<int> hidden_nodes,int output_nodes) {
        int prev_nodes = input_nodes;
        for(int i=0;i<hidden_nodes.size();i++){
            vector<size_t> dims = {hidden_nodes[i],prev_nodes};
            auto t = Tensor::random(dims);
            this->weights.push_back(t);
            prev_nodes = hidden_nodes[i];
        }
        vector<size_t> dims = {output_nodes,prev_nodes};
        auto t = Tensor::random(dims);
        this->weights.push_back(t);
    }

    void initializeBiases(vector<int> hidden_nodes,int output_nodes) {
        for(int i=0;i<hidden_nodes.size();i++){
            vector<size_t> dims = {hidden_nodes[i],1};
            auto t = Tensor::random(dims);
            biases.push_back(t);
        }
        vector<size_t> dims = {output_nodes,1};
        Tensor t = Tensor::random(dims);
        biases.push_back(t);
    }

    void initializeActivations(int input_nodes,vector<int> hidden_nodes,int output_nodes) {
        activations.push_back(Tensor({input_nodes,1},true));
        linear_activations.push_back(Tensor({input_nodes,1},true));
        for(int i=0;i<hidden_nodes.size();i++){
            activations.push_back(Tensor({hidden_nodes[i],1},true));
            linear_activations.push_back(Tensor({hidden_nodes[i],1},true));
        }
        activations.push_back(Tensor({output_nodes,1},true));
        linear_activations.push_back(Tensor({output_nodes,1},true));
    }

    Tensor forward(Tensor input) {

        activations[0] = activations[0].add(input);
        linear_activations[0] = linear_activations[0].add(input);
        for(int i=0;i<weights.size();i++){
            Tensor z = weights[i].matmul(activations[i]).add(biases[i]);
            Tensor a = z.applyActivation(activation_functions[i]);
            activations[i+1] = activations[i+1].add(a);
            linear_activations[i+1] = linear_activations[i+1].add(z);
        }
        return activations[activations.size()-1];
    }

    Tensor forwardSet(Tensor input) {

        activations[0] = input;
        linear_activations[0] = input;
        for(int i=0;i<weights.size();i++){
            Tensor z = weights[i].matmul(activations[i]).add(biases[i]);
            Tensor a = z.applyActivation(activation_functions[i]);
            activations[i+1] = a;
            linear_activations[i+1] = z;
        }

        return activations[activations.size()-1];
    }


    double lossMSE(Tensor output,Tensor target) {
        Tensor error = output.subtract(target);
        Tensor squared_error = error.elementwise_mult(error);
        double loss = 0;
        squared_error.sync();
        for(auto x : squared_error.data)
            loss += x;
        return loss/(2.0*batch_size);
    }

    double lossCrossEntropy(Tensor output,Tensor target) {
        Tensor error = output.subtract(target);
        Tensor log_error = error.log();
        double loss = 0;
        log_error.sync();
        for(auto x : log_error.data)
            loss += x;
        return -loss/batch_size;
    }

    Tensor MSEDerivative(Tensor output,Tensor target,int batch_size) {
        Tensor error = output.subtract(target);
        return error.mult(1.0/batch_size);
    }

    Tensor CrossEntropyDerivative(Tensor output,Tensor target,int batch_size) {
        Tensor error = output.subtract(target);
        return error.mult(1.0/batch_size);
    }

    Tensor getLossDerivative(Tensor output,Tensor target) {
        if(loss_function == MSE)
            return MSEDerivative(output,target,output.dims[0]);
        else if(loss_function == CROSS_ENTROPY)
            return CrossEntropyDerivative(output,target,output.dims[0]);
        else
            throw "Unknown loss function";
    }

    double getLoss(Tensor output,Tensor target) {
        if(loss_function == MSE)
            return lossMSE(output,target);
        else if(loss_function == CROSS_ENTROPY)
            return lossCrossEntropy(output,target);
        else
            throw "Unknown loss function";
    }

    void backward(Tensor output,Tensor target,double learning_rate) {
        if(show_loss){
            double loss = getLoss(output,target);
            accumulated_loss += loss;
            if(curr_processed%batch_size==0){
                cout<<"Loss: "<<accumulated_loss<<endl;
                accumulated_loss = 0;
            }
        }
        
        Tensor error = getLossDerivative(output,target);
        if(loss_function == CROSS_ENTROPY){
            //Apply softmax derivative
            Tensor a = activations[activations.size()-1];
            double min_val=a.min();
            Tensor ones = Tensor::ones({a.dims[1],1});
            Tensor min_tensor=ones.mult(min_val);
            a = a.subtract(min_tensor);
            Tensor exponent = a.exp();
            Tensor sum = exponent.matmul(ones);
            Tensor softmax_derivative = exponent.divided_by(sum);
            error = error.elementwise_mult(softmax_derivative);
        }
        // error.print();
        Tensor prev_error = error;
        for(int i=weights.size()-1;i>=0;i--){
            Tensor a = activations[i];
            Tensor z = linear_activations[i+1];
            Tensor dz = z.applyDerivative(activation_functions[i]);
            Tensor delta = prev_error.elementwise_mult(dz);
            Tensor a_t = a.transpose();
            Tensor weight_gradient = delta.matmul(a_t);
            Tensor bias_gradient = delta;
            Tensor weight_update = weight_gradient.mult(-learning_rate);
            Tensor bias_update = bias_gradient.mult(-learning_rate);
            weights[i] = weights[i].add(weight_update);
            biases[i] = biases[i].add(bias_update);
            Tensor w_t = weights[i].transpose();
            prev_error = w_t.matmul(delta);
        }
    }

    void train(vector<vector<double>> inputs,vector<vector<double>> targets,int epochs,double learning_rate) {

        if(inputs.size() != targets.size())
            throw "Mismatched number of inputs and targets.\n Expected: "+to_string(inputs.size())+"\n Found: "+to_string(targets.size());
        if(inputs[0].size()!=weights[0].dims[1])
            throw "Mismatched input size.\n Expected: "+to_string(weights[0].dims[1])+"\n Found: "+to_string(inputs[0].size());
        if(targets[0].size()!=weights[weights.size()-1].dims[0])
            throw "Mismatched target size.\n Expected: "+to_string(weights[weights.size()-1].dims[0])+"\n Found: "+to_string(targets[0].size());
        // cout<<"Uploading data..."<<endl;
        vector<size_t> input_dims = {inputs[0].size(),1};
        vector<size_t> target_dims = {targets[0].size(),1};
        vector<Tensor> input_tensors;
        vector<Tensor> target_tensors;
        for(int i=0;i<inputs.size();i++){
            Tensor input(input_dims,true);
            input.data = inputs[i];
            input.upload();
            input_tensors.push_back(input);
            Tensor target(target_dims,true);
            target.data = targets[i];
            target.upload();
            target_tensors.push_back(target);
        }
        // cout<<"Beginning training..."<<endl;
        for(int i=0;i<epochs;i++){
            for(int j=0;j<inputs.size();j++){
                Tensor input = input_tensors[j];
                Tensor output;
                if(curr_processed%batch_size==0)
                    output = forwardSet(input);
                else
                    output = forward(input);
                curr_processed++;
                Tensor target = target_tensors[j];
                backward(output,target,learning_rate);
            }
        }
    }

    Tensor predict(vector<double> input) {
        if(input.size()!=weights[0].dims[1])
            throw "Mismatched input size.\n Expected: "+to_string(weights[0].dims[1])+"\n Found: "+to_string(input.size());
        vector<size_t> input_dims = {input.size(),1};
        Tensor input_tensor(input_dims,true);
        input_tensor.data = input;
        input_tensor.upload();
        return forwardSet(input_tensor);
    }

    void saveModel(string filename) {
        for(int i=0;i<weights.size();i++){
            weights[i].sync();
            biases[i].sync();
        }
        ofstream file(filename);
        for(int i=0;i<weights.size();i++){
            for(auto x : weights[i].data)
                file<<x<<" ";
            file<<endl;
            for(auto x : biases[i].data)
                file<<x<<" ";
            file<<endl;
        }
        file.close();
    }

    void loadModel(string filename) {
        ifstream file(filename);
        for(int i=0;i<weights.size();i++){
            for(int j=0;j<weights[i].data.size();j++)
                file>>weights[i].data[j];
            for(int j=0;j<biases[i].data.size();j++)
                file>>biases[i].data[j];
            weights[i].upload();
            biases[i].upload();
        }
        file.close();
    }

    void print() {
        for(int i=0;i<weights.size();i++){
            cout<<"Weights: "<<i+1<<endl;
            weights[i].print();
            cout<<"Biases: "<<i+1<<endl;
            biases[i].print();
        }
    }

};


// int main() {
//     try{
//         NeuralNetwork nn(2,{2},1,SIGMOID);
//         nn.setLossFunction(MSE);
//         //Test on XOR dataset
//         vector<vector<double>> inputs = {{0,0},{0,1}};
//         vector<vector<double>> targets = {{0},{1}};
//         nn.train(inputs,targets,10000,0.1);
//         for(auto input : inputs){
//             Tensor output = nn.predict(input);
//             output.sync();
//             cout<<"Input: ";
//             for(auto x : input)
//                 cout<<x<<" ";
//             cout<<"Output: ";
//             for(auto x : output.data)
//                 cout<<x<<" ";
//             cout<<endl;
//         }
//     } catch(string e){
//         cout<<e<<endl;
//     } catch(const char* e){
//         cout<<e<<endl;
//     }
// }