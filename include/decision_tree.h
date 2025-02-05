#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <string>
#include <fstream>
#include <omp.h>
using namespace std;

class DecisionTree {
        public:

        struct Node{
            bool isLeaf;
            int featureIndex;
            double threshold;
            string classLabel;
            Node* left;
            Node* right;

            Node():
                isLeaf(false),
                featureIndex(0),
                threshold(0.0),
                left(nullptr),
                right(nullptr)
            {}
        };

        Node* root;
        const int MAX_DEPTH = 3;
        int level = 1;


        // Constructor
        DecisionTree();
        
        // Method to compute entropy
        double entropy(vector<string>& labels);

        // Method to train the decision tree
        void train(vector<vector<double>>& data, vector<string>& labels);

        // Helper method to train the decision tree
        void _train(vector<vector<double>>& data, vector<string>& labels, Node* node,int level);

        // Method to predict based on trained tree
        vector<string> predict(vector<vector<double>>& data);

        string predictSingle(vector<double>& data);

        // Helper method to predict based on trained tree
        string _predict(vector<double>& data, Node* node);

        // Method to test the accuracy of the decision tree
        double test(vector<vector<double>>& data, vector<string>& labels);

        // Method to split a CSV line into values and store labels
        vector<double> split(string line, char delimiter, vector<string>& labels);

        // Method to load data from CSV file
        void loadData(string filename, vector<vector<double>>& data, vector<string>& labels);

        // Method to save the model to a file
        void saveModel(string filename);

        // Method to load the model from a file
        void loadModel(string filename);

        // Helper method to save the model to a file
        void _saveModel(Node* node, ofstream& file);

        // Helper method to load the model from a file
        void _loadModel(Node* node, ifstream& file);
};
