#ifndef CUDA_H
#define CUDA_H

#include <cuda_runtime.h>

void find_best_split(double* data, int size,int cols,double* uniqueAverages,int uniqueAvgSize,double* gains,
int gainsSize, double* labels,int labelsSize, int featureIndex, double* leftLabels,double* rightLabels, double* uniqueLabels);
#endif
