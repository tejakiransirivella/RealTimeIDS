#include<iostream>
#include<cuda_runtime.h>
#include "cuda.h"
#include <assert.h>
#include<stdio.h>
using namespace std;


/**
 * @brief This function calculates the entropy of the labels
 * @param labels: The labels of the data
 * @param labelsSize: The size of the labels
 * @param uniqueLabels: The unique labels
 * @return The entropy of the labels
 */
__device__ double entropy(double* labels, int labelsSize,double* uniqueLabels){
    double entropy = 0;
    for(int i = 0; i < labelsSize; i++){
        uniqueLabels[i] = 0;
    }
    for(int i = 0; i < labelsSize; i++){
        uniqueLabels[(int)labels[i]] += 1;
    }

    for(int i = 0; i < labelsSize; i++){
        if(uniqueLabels[i] == 0){
            continue;
        }
        uniqueLabels[i] /= labelsSize;
        entropy += -uniqueLabels[i] * log2(uniqueLabels[i]);
    }

    return entropy;
}

/**
 * @brief This is the kernel function that calculates the information gain for each threshold
 */

__global__ void find_best_split_kernel(double* data, int size,int cols,double* uniqueAverages,
int uniqueAvgSize,double* gains,int gainsSize, double* labels,int labelsSize, int featureIndex,
double* leftLabels,double* rightLabels,double* uniqueLabels){
    int x = threadIdx.x + (blockIdx.x * blockDim.x);
    if(x < uniqueAvgSize){
      double threshold = uniqueAverages[x];
    
      int leftLabelsSize = 0;
      int rightLabelsSize = 0;
        for(int i = 0; i < labelsSize; i++){ 
            if(data[i * cols + featureIndex] < threshold){
                leftLabels[leftLabelsSize] = labels[i];
                leftLabelsSize++;
            }else{
                rightLabels[rightLabelsSize] = labels[i];
                rightLabelsSize++;
            }
        }

        double leftEntropy = entropy(leftLabels, leftLabelsSize,uniqueLabels);
        double rightEntropy = entropy(rightLabels, rightLabelsSize,uniqueLabels);
        double pLeft = (double)leftLabelsSize / labelsSize;
        double pRight = (double)rightLabelsSize / labelsSize;
        gains[x] = entropy(labels, labelsSize,uniqueLabels) - ((pLeft * leftEntropy) + (pRight * rightEntropy));
        
    }
}



/**
 * @brief This function finds the best split for a feature
 */
void find_best_split(double* data, int size,int cols,double* uniqueAverages,int uniqueAvgSize,double* gains,
int gainsSize, double* labels,int labelsSize, int featureIndex,double* leftLabels,double* rightLabels,double* uniqueLabels){

    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    
    int threadsPerBlock = 128;
    int blockPerGrid = (uniqueAvgSize + threadsPerBlock - 1)/ threadsPerBlock;

    find_best_split_kernel<<<blockPerGrid,threadsPerBlock>>>(data, size,cols, uniqueAverages, 
    uniqueAvgSize, gains, gainsSize, labels, labelsSize,featureIndex,leftLabels,rightLabels,uniqueLabels);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // cudaDeviceSynchronize();
}


