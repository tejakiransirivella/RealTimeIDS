#pragma once
#include <map>
#include <vector>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <memory>
#include "timer.h"
class MemoryManager{
public:
  std::map<int,double*> tensorGpuDataMap;
  std::map<int,int> tensorSizeMap, tensorRefCountMap;
  std::vector<int> destructIds;
  std::vector<std::vector<double>> scheduledUploads;
  std::map<int, bool> hasScheduledUploads;
  int destructSize = 0; 
  
  // double uploadTime = 0;
  // int uploadCount=0;
  // double maxUploadTime = 0;
  void uploadData(int id, std::vector<double> &data) {
    // Timer t;
    // t.restart();
    double *d_data;
    
    cudaError_t err = cudaMalloc((void **)&d_data,data.size()*sizeof(double));
    if (err != cudaSuccess) {
      std::cerr << "Error allocating memory on GPU: " << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("CUDA allocation error");
    }
    err = cudaMemcpy(d_data,data.data(),data.size()*sizeof(double),cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cerr << "Error copying data to GPU: " << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("CUDA copy error");
    }
    tensorGpuDataMap[id] = d_data;
    tensorSizeMap[id] = data.size();
    // auto timeTaken=t.get_time();
    // uploadTime+=timeTaken;
    // uploadCount++;
    // if(timeTaken > maxUploadTime){
    //   maxUploadTime = timeTaken;
    // }

  }

  void scheduleDataUpload(int id,std::vector<double> &data) {
    std::vector<double> dataCopy = data; // Create a copy of the data
    scheduledUploads.push_back(dataCopy);
    hasScheduledUploads[id] = true;
  }

  void processScheduledUploads() {
    if(scheduledUploads.size() == 0) return;
    int i=0;
    int sz=scheduledUploads.size();
    double *d_data[sz];
    cudaStream_t stream;
    cudaError_t status = cudaStreamCreate(&stream);
    if (status != cudaSuccess) {
      std::cerr << "Error creating stream: " << cudaGetErrorString(status) << std::endl;
      throw std::runtime_error("CUDA stream creation error");
    }
    for (; i < sz; i++ ) {
      cudaError_t err = cudaMallocAsync((void **)&d_data[i],scheduledUploads[i].size()*sizeof(double),stream);
      if (err != cudaSuccess) {
        std::cerr << "Error allocating memory on GPU: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA allocation error");
      }
      err = cudaMemcpyAsync(d_data[i],scheduledUploads[i].data(),scheduledUploads[i].size()*sizeof(double),cudaMemcpyHostToDevice,stream);
      if (err != cudaSuccess) {
        std::cerr << "Error copying data to GPU: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA copy error");
      }
      
    }
    cudaStreamSynchronize(stream);
    i=0;
    for (const auto& [id,_] : hasScheduledUploads) {
      tensorGpuDataMap[id] = d_data[i];
      tensorSizeMap[id] = scheduledUploads[i].size();
      i++;
    }

    scheduledUploads.resize(0);
    hasScheduledUploads.clear();
    cudaStreamDestroy(stream);
  }
  bool exists(int id){
    return tensorGpuDataMap.find(id) != tensorGpuDataMap.end();
  }

  void allocateData(int id, int size){
    double *d_data;
    cudaMalloc(&d_data,size*sizeof(double));
    tensorGpuDataMap[id] = d_data;
    tensorSizeMap[id] = size;
  }

  void downloadData(int id, std::vector<double> &data){

    double *d_data = tensorGpuDataMap[id];
    data.resize(tensorSizeMap[id]);
    cudaMemcpy(data.data(),d_data,data.size()*sizeof(double),cudaMemcpyDeviceToHost);

  }

  void addReference(int id){
    tensorRefCountMap[id]++;
  }

  void removeReference(int id){
    tensorRefCountMap[id]--;
    if(tensorRefCountMap[id] == 0){
      destructIds.push_back(id);
      // destructSize+=tensorSizeMap[id]*sizeof(double);
      // if(destructSize > 1024*1024*1024){ // 1GB
      //   printf("Size of destructIds: %d\n", destructIds.size());
      //   printf("Size of destructSize: %d\n", destructSize);
      //   for(size_t i = 0; i < destructIds.size(); i++){
      //     freeData(destructIds[i]);
      //   }
      //   destructIds.resize(0);
      //   destructSize = 0;
      // }
    }
    
  }

  void freeData(int id){
    double *d_data = tensorGpuDataMap[id];
    cudaFree(d_data);
    tensorGpuDataMap.erase(id);
  }

  double* getGpuData(int id){
    return tensorGpuDataMap[id];
  }

  int getTensorSize(int id){
    return tensorSizeMap[id];
  }

  ~MemoryManager(){
    for(auto &pair : tensorGpuDataMap){
      cudaFree(pair.second);
    }
  }
};



extern MemoryManager memoryManager; // Global instance of MemoryManager
