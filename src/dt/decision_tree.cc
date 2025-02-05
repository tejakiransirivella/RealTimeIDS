#include<string.h>
#include "../../include/decision_tree.h"
#include<cuda_runtime.h>
#include "../../include/cuda.h"


/**
 * @brief Construct a new deicion tree object with the root node
 */
DecisionTree::DecisionTree() {
    root = new Node();
    level = 0;
    
}

/**
 * @brief Calculate the entropy of a set of labels
 * @param labels The set of labels
 * @return The entropy of the set of labels
 */
double DecisionTree::entropy(vector<string>& labels) {
    double ent = 0;
    map<string, double> labelCount;

    for(int i = 0; i < labels.size(); i++) {
        labelCount[labels[i]]++;
    }

    for(const auto& pair : labelCount) {
        double prob = pair.second / labels.size();
        if (prob > 0) {
            ent -= prob * log2(prob);
        }
    }

    return ent;
}


/**
 * @brief Train the decision tree
 * @param data The training data
 * @param labels The labels of the training data
 */
void DecisionTree::train(vector<vector<double>>& data, vector<string>& labels) {
    _train(data, labels, root,1);
}


/**
 * @brief Train the decision tree
 * @param data The training data
 * @param labels The labels of the training data
 * @param node The current node
 * @param level The current level of the tree
 */
void DecisionTree::_train(vector<vector<double>>& data, vector<string>& labels, Node* node, int level) {
        double nodeEntropy = entropy(labels);
        /*
        * If the data is empty, the entropy is 0, or the tree has reached the maximum depth, the node is a leaf node
        */
        if(data.size() <= 1 || nodeEntropy == 0 || level == MAX_DEPTH){
            if(level < MAX_DEPTH){
                node->isLeaf = true;
                node->classLabel = labels[0];
            }
            else{
                map<string, double> labelCount;
                for(int i = 0; i < labels.size(); i++){
                    labelCount[labels[i]]++;
                }
                string maxLabel;
                double maxCount = 0;
                for(const auto& pair : labelCount){
                    if(pair.second > maxCount){
                        maxCount = pair.second;
                        maxLabel = pair.first;
                    }
                }
                node->isLeaf = true;
                node->classLabel = maxLabel;
            }
            return;
        }

        double maxGain = 0;
        int bestFeatureIndex = 0;
        double bestThreshold = 0;

        
        /*
        * Flatten the data to a 1D array , map the unique labels to indices and create a vector of label indices
        */
        std::vector<double> flattened_data;
        for(const auto& row : data){
            flattened_data.insert(flattened_data.end(),row.begin(),row.end());
        }

        double* data_d;
        cudaMalloc(&data_d,flattened_data.size()*sizeof(double));
        cudaMemcpy(data_d,flattened_data.data(),flattened_data.size()*sizeof(double),cudaMemcpyHostToDevice);
        
        double* labels_d;
        cudaMalloc(&labels_d,labels.size()*sizeof(double));

        set<string> uniqueLabels(labels.begin(), labels.end());
        vector<double> labelIndices;
        map<string, double> labelIndexMap;

        int index = 0;
        for(const auto& label : uniqueLabels){
            labelIndexMap[label] = index;
            index++;
        }

        for(const auto& label : labels){
            labelIndices.push_back(labelIndexMap[label]);
        }

       cudaMemcpy(labels_d,labelIndices.data(),labelIndices.size()*sizeof(double),cudaMemcpyHostToDevice);


        /**
         * For each feature, find the best threshold to split the data on
         */

        #pragma omp parallel for
        for(int i = 1; i < data[0].size(); i++){
            set<double> uniqueValues;
            for(int j = 0; j < data.size(); j++){
                uniqueValues.insert(data[j][i]);
            }

            vector<double> uniqueValuesVec(uniqueValues.begin(), uniqueValues.end());
            vector<double> uniqueAverages;
            for(int k = 0; k < uniqueValuesVec.size()-1; k++){
                uniqueAverages.push_back((uniqueValuesVec[k] + uniqueValuesVec[k+1]) / 2);
            }

            if(uniqueAverages.size() == 0){
                continue;
            }

            double* uniqueValues_d;
            cudaMalloc(&uniqueValues_d,uniqueAverages.size()*sizeof(double));
            cudaMemcpy(uniqueValues_d,uniqueAverages.data(),uniqueAverages.size()*sizeof(double),cudaMemcpyHostToDevice);

            double* gains_d;
            cudaMalloc(&gains_d,uniqueAverages.size()*sizeof(double));

            double* leftLabels_d;
            double* rightLabels_d;
            double* uniqueLabels_d;
            cudaMalloc(&leftLabels_d,labels.size()*sizeof(double));
            cudaMalloc(&rightLabels_d,labels.size()*sizeof(double));
            cudaMalloc(&uniqueLabels_d,labels.size()*sizeof(double));

            find_best_split(data_d,flattened_data.size(),data[0].size(),uniqueValues_d,uniqueAverages.size(),gains_d,
            uniqueAverages.size(),labels_d,labels.size(),i,leftLabels_d,rightLabels_d,uniqueLabels_d);

            double* gains = new double[uniqueAverages.size()];
            cudaMemcpy(gains,gains_d,uniqueAverages.size()*sizeof(double),cudaMemcpyDeviceToHost);

            // Find the best gain
            for(int k = 0; k < uniqueAverages.size(); k++){
                #pragma omp critical
                {
                    if(gains[k] > maxGain){
                        maxGain = gains[k];
                        bestFeatureIndex = i;
                        bestThreshold = uniqueAverages[k];
                    }
                }
            }

            cudaFree(uniqueValues_d);
            cudaFree(gains_d);
            cudaFree(leftLabels_d);
            cudaFree(rightLabels_d);
            cudaFree(uniqueLabels_d);
             
        }

        node->featureIndex = bestFeatureIndex;
        node->threshold = bestThreshold;

        vector<vector<double>> leftData;
        vector<string> leftLabels;
        vector<vector<double>> rightData;
        vector<string> rightLabels;
        for(int i = 0; i < data.size(); i++){
            if(data[i][bestFeatureIndex] < bestThreshold){
                leftData.push_back(data[i]);
                leftLabels.push_back(labels[i]);
            } else {
                rightData.push_back(data[i]);
                rightLabels.push_back(labels[i]);
            }
        }

      
        if(leftData.size() == 0 || rightData.size() == 0){
            node->isLeaf = true;
            node->classLabel = labels[0];
            return;
        }
        node->left = new Node();
        node->right = new Node();
        _train(leftData, leftLabels, node->left,level+1);
        _train(rightData, rightLabels, node->right,level+1);
}

/**
 * @brief Predict the labels of a set of data
 * @param data The data to predict
 * @return The predicted labels
 */
vector<string> DecisionTree::predict(vector<vector<double>>& data) {
    vector<string> predictions;
    for(int i = 0; i < data.size(); i++) {
        predictions.push_back(predictSingle(data[i]));
    }
    return predictions;
}

/**
 * @brief Predict the label of a single data point
 * @param data The data point to predict
 * @return The predicted label
 */
string DecisionTree::predictSingle(vector<double>& data){
    return _predict(data, root);
}

/**
 * @brief Predict the label of a single data point
 * @param data The data point to predict
 * @param node The current node
 * @return The predicted label
 */
string DecisionTree::_predict(vector<double>& data, Node* node) {
    if(node->isLeaf) {
        return node->classLabel;
    }

    if(data[node->featureIndex] < node->threshold) {
        return _predict(data, node->left);
    } else {
        return _predict(data, node->right);
    }
}


/**
 * @brief Test the decision tree
 * @param data The test data
 * @param labels The labels of the test data
 * @return The accuracy of the decision tree
 */
double DecisionTree::test(vector<vector<double>>& data, vector<string>& labels) {
    int correct = 0;
    vector<string> predictions = predict(data);
    for(int i = 0; i < data.size(); i++) {
        if(predictions[i] == labels[i]) {
            correct++;
        }
    }
    return (double)correct / data.size();
}


/**
 * @brief Split a data point string by a delimiter (",") to convert to a vector
 * @param line single data point string
 * @param delimiter The delimiter to split by
 * @param labels The labels of the data
 * @return The data point as a vector
 */
vector<double> DecisionTree::split(string line, char delimiter, vector<string>& labels) {
    vector<double> row;
    string value = "";

    for(int i = 0; i < line.size(); i++) {
        if(line[i] == delimiter) {
            row.push_back(stod(value));
            value = "";
        } else {
            value += line[i];
        }
    }

    labels.push_back(value);
    return row;
}

/**
 * @brief Load the data from a file
 * @param filename The name of the file
 * @param data The data to load
 * @param labels The labels of the data
 */
void DecisionTree::loadData(string filename, vector<vector<double>>& data, vector<string>& labels) {
    cout << "Loading data" << endl;
    ifstream file(filename);

    if(file.is_open()) {
        string line;
        getline(file, line);
        while(getline(file, line)) {
            vector<double> row = split(line, ',', labels);
            data.push_back(row);
        }
        file.close();
    } else {
        cout << "Error opening file" << endl;
    }
    cout << "Data loaded" << endl;
}

/**
 * @brief Save the decision tree model to a file by recursively saving the nodes
 * @param node The current node initially the root
 */
void DecisionTree::_saveModel(Node* node, ofstream& file){
    if(node->isLeaf){
        file << "L," << node->classLabel << endl;
        return;
    }
    file << "I," << node->featureIndex << "," << node->threshold << endl;
    _saveModel(node->left, file);
    _saveModel(node->right, file);
}

/**
 * @brief Load the decision tree model from a file by recursively loading the nodes
 */
void DecisionTree::_loadModel(Node* node, ifstream& file){
    string line;
    getline(file, line);
    if(line[0] == 'L'){
        node->isLeaf = true;
        node->classLabel = line.substr(2);
        return;
    }
    node->isLeaf = false;
    int commaIndex = line.find(',');
    node->featureIndex = stoi(line.substr(commaIndex+1, line.find(',', commaIndex+1)));
    node->threshold = stod(line.substr(line.find(',', commaIndex+1)+1));
    node->left = new Node();
    node->right = new Node();
    _loadModel(node->left, file);
    _loadModel(node->right, file);
}

/**
 * @brief Save the decision tree model to a file
 */
void DecisionTree::saveModel(string filename){
    ofstream file(filename);
    _saveModel(root, file);
    file.close();
}

/**
 * @brief Load the decision tree model from a file
 */
void DecisionTree::loadModel(string filename){
    ifstream file(filename);
    _loadModel(root, file);
}


int main(){
    vector<vector<double>> data;
    vector<string> labels;
    DecisionTree* dt = new DecisionTree();
    dt->loadData("CICIDS2017_sample.csv", data, labels);
    dt->train(data, labels);
    dt->saveModel("tree.model");
    cout << "accuracy: " << (dt->test(data, labels))*100 << endl;
    return 0;
}
