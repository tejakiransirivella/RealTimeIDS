#pragma once
#include<vector>
#include<omp.h>
#include<map>
#include<cuda_runtime.h>
#include<random>
#include "memory_manager.h"
#include "gpu.h"
#include "timer.h"
#include "utils.h"
// #pragma omp parallel num_threads(8)

enum ActivationFunction {
    SIGMOID,
    RELU,
    SOFTMAX
};


class Tensor{
public:
  std::vector<double> data;
  std::vector<size_t> dims;
  GPUExecutor gpuExecutor;
  int id;
  bool onlyGPU,transposed;
  static int count;
  Tensor() : dims({0}),onlyGPU(true),transposed(false){
    id = count;
    ++count;
    memoryManager.allocateData(id,0);
    memoryManager.addReference(id);
  }

  Tensor(std::vector<size_t> dims,bool onlyGPU=false,bool transposed=false) : dims(dims),onlyGPU(onlyGPU),transposed(transposed){
    id = count;
    ++count;
    size_t len = 1;
    for(auto d : dims)
      len *= d;
    if(!onlyGPU){
      data.reserve(len);
      data.resize(len);
    }
    memoryManager.allocateData(id,len);
    memoryManager.addReference(id);
  }

  Tensor(Tensor& x,std::vector<size_t> dims) : dims(dims),onlyGPU(x.onlyGPU),transposed(false){
    id = x.id;
    if(!onlyGPU){
      data = x.data;
    }
    memoryManager.addReference(id);

  }

  Tensor(std::vector<size_t> dims,std::vector<std::vector<size_t>> idx,std::vector<double> val) : dims(dims),onlyGPU(false),transposed(false){
    id = count;
    ++count;
    size_t len = 1;
    for(auto d : dims)
      len *= d;
    data.resize(len);
    if(idx.size() != val.size())
      throw "Mismatched idx and val size";
    for(size_t i = 0;i < idx.size();++i){
      data[index(idx[i])] = val[i];
    }

    memoryManager.scheduleDataUpload(id,data);
    memoryManager.addReference(id);

  }

  static Tensor ones(std::vector<size_t> dims){
    Tensor ret(dims);
    for(size_t i = 0;i < ret.data.size();++i)
      ret.data[i] = 1;
    ret.upload();
    return ret;
  }

  static Tensor random(std::vector<size_t> dims){
    Tensor ret(dims);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0,2);
    for(size_t i = 0;i < ret.data.size();++i)
      ret.data[i] = d(gen);
    ret.upload();
    return ret;
  }

  size_t index(std::vector<size_t> x){
    if(x.size() != dims.size())
      throw "Mismatched dims in index";
    size_t ret = 0;
    size_t prod = 1;
    for(int i = dims.size() - 1;i >= 0;--i){
      if(x[i] >= dims[i])
        throw "Index out of bound";
      ret += x[i] * prod;
      prod *= dims[i];
    } 
    return ret;
  }

  void ensureSafeAccess(){
    
    if(__builtin_expect(memoryManager.hasScheduledUploads.find(id) != memoryManager.hasScheduledUploads.end(),0)){
      memoryManager.processScheduledUploads();
    }
  }

  Tensor reshape(std::vector<size_t> new_dims){
    ensureSafeAccess();
    size_t len = 1;
    for(auto d : new_dims)
      len *= d;
    if(onlyGPU){
      if(len != (size_t)memoryManager.getTensorSize(id))
        throw "Mismatched dims in reshape";
      Tensor ret(*this,new_dims);
      return ret;
    }else{
      if(len != data.size())
        throw "Mismatched dims in reshape";
      Tensor ret(*this,new_dims);
      return ret;
    }
  }

  Tensor transpose(){
    ensureSafeAccess();

    if(dims.size() == 2){
      Tensor ret({dims[1],dims[0]},true);
      gpuExecutor.launchTranspose2d(id,ret.id,dims[0],dims[1]);
      
      return ret;
    }else if(dims.size() == 3){
      Tensor ret({dims[0],dims[2],dims[1]},true);
      gpuExecutor.launchTranspose3d(id,ret.id,dims[1],dims[2],dims[0]);
      return ret;
    }else{
      throw "The tensor must be 2D or batched 2D tensors";
    }

  }

  Tensor neg(){
    ensureSafeAccess();

    Tensor ret(dims,true);
    gpuExecutor.launchNegate(id,ret.id);
    return ret;
  }
  
  Tensor reciprocal(){
    ensureSafeAccess();

    Tensor ret(dims,true);
    gpuExecutor.launchReciprocal(id,ret.id);
    return ret;
  }

  Tensor add(Tensor& x){
    ensureSafeAccess();
    x.ensureSafeAccess();

    if(dims != x.dims)
      throw "Mismatched shape in add";
    Tensor ret(dims,true);
    gpuExecutor.launchAdd(id,x.id,ret.id);
    return ret;
  }
  
  Tensor subtract(Tensor& x){
    ensureSafeAccess();
    x.ensureSafeAccess();

    if(dims != x.dims)
      throw "Mismatched shape in subtract";
    Tensor ret(dims,true);
    gpuExecutor.launchSubtract(id,x.id,ret.id);
    return ret;
  }

  Tensor mult(double x){
    ensureSafeAccess();

    Tensor ret(dims,true);
    gpuExecutor.launchMultiply(id,x,ret.id);
    return ret;
  }
  
  Tensor elementwise_mult(Tensor& x){
    ensureSafeAccess();
    x.ensureSafeAccess();

    if(dims != x.dims)
      throw "Mismatched shape in elementwise_mult";
    Tensor ret(dims,true);
    gpuExecutor.launchElementwiseMultiply(id,x.id,ret.id);
    return ret;
  }
  
  Tensor pow(double x){
    ensureSafeAccess();

    Tensor ret(dims,true);
    gpuExecutor.launchPower(id,x,ret.id);
    return ret;
  }
  
  Tensor relu(){
    ensureSafeAccess();

    Tensor ret(dims,true);
    gpuExecutor.launchRelu(id,ret.id);
    return ret;
  }

  Tensor binarilize(){
    ensureSafeAccess();

    Tensor ret(dims,true);
    gpuExecutor.launchBinarilize(id,ret.id);
    return ret;
  }

  Tensor exp(){
    ensureSafeAccess();

    Tensor ret(dims,true);
    gpuExecutor.launchExp(id, ret.id);
    return ret;
  }

  Tensor sigmoid(){
    ensureSafeAccess();
    Tensor ret(dims,true);
    gpuExecutor.launchSigmoid(id, ret.id);
    return ret;
  }

  double min(){
    ensureSafeAccess();
    sync();
    double ret = data[0];
    for(auto x : data)
      ret = std::min(ret,x);
    return ret;
  }

  Tensor divided_by(Tensor& x){
    ensureSafeAccess();
    x.ensureSafeAccess();
    if(dims != x.dims)
      throw "Mismatched shape in divided_by";
    Tensor ret(dims,true);
    gpuExecutor.launchDivide(id,x.id,ret.id);
    return ret;
  }

  Tensor matmul(Tensor& x){

    ensureSafeAccess();
    x.ensureSafeAccess();
    if(x.dims.size() != 2){
      printVector(x.dims);
      throw "The right operand of matmul must be 2D tensors. Found " + std::to_string(x.dims.size()) + "D tensor";
    }

    if(dims.size() != 2 && dims.size() != 3){
      throw "The left operand of matmul must be 2D tensors or batched 2D tensors";
    }

    if(dims[dims.size() - 1] != x.dims[0]){
      throw "Mismatched matmul matrix dimensions";
    }
    if(dims.size() == 2){
      Tensor ret({dims[0],x.dims[1]},true);
      gpuExecutor.launchMatmul(id,x.id,ret.id,dims[0],x.dims[1],dims[1],1);
      return ret;
    }else{
      Tensor ret({dims[0],dims[1],x.dims[1]},true);
      gpuExecutor.launchMatmul(id,x.id,ret.id,dims[1],x.dims[1],dims[2],dims[0]);
      return ret;
    }
    //   #pragma omp parallel for
    //   for(size_t i = 0;i < dims[0];++i){
    //     for(size_t j = 0;j < x.dims[1];++j){
    //       for(size_t k = 0;k < dims[1];++k){
    //         ret.data[ret.index({i,j})] += data[index({i,k})] * x.data[x.index({k,j})];
    //       }
    //     }
    //   }
    //   return ret;
    // }else{
    //   Tensor ret({dims[0],dims[1],x.dims[1]});
    //   #pragma omp parallel for
    //   for(size_t b = 0;b < dims[0];++b){
    //     for(size_t i = 0;i < dims[1];++i){
    //       for(size_t j = 0;j < x.dims[1];++j){
    //         for(size_t k = 0;k < dims[2];++k){
    //           ret.data[ret.index({b,i,j})] += data[index({b,i,k})] * x.data[x.index({k,j})];
    //         }
    //       }
    //     }
    //   }
    //   return ret;
    // }
  }

  Tensor log(){
    ensureSafeAccess();
    Tensor ret(dims,true);
    gpuExecutor.launchLog(id,ret.id);
    return ret;
  }

  Tensor applyActivation(ActivationFunction activation){
    ensureSafeAccess();
    Tensor ret(dims,true);
    if(activation == SIGMOID){
      gpuExecutor.launchSigmoid(id,ret.id);
    }else if(activation == RELU){
      gpuExecutor.launchRelu(id,ret.id);
    }else if(activation == SOFTMAX){
      Tensor exp = this->exp();
      Tensor ones = Tensor::ones({dims[1],1});
      Tensor sum = exp.matmul(ones);
      Tensor reciprocal_sum = sum.reciprocal();
      
      ret = exp.elementwise_mult(reciprocal_sum);
    }else{
      throw "Unknown activation function";
    }
    return ret;
  }

  Tensor applyDerivative(ActivationFunction activation){
    ensureSafeAccess();
    Tensor ret(dims,true);
    if(activation == SIGMOID){
      gpuExecutor.launchSigmoidDerivative(id,ret.id);
    }else if(activation == RELU){
      gpuExecutor.launchReluDerivative(id,ret.id);
    }else if(activation == SOFTMAX){
      gpuExecutor.launchSoftmaxDerivative(id,ret.id);
    }else{
      throw "Unknown activation function";
    }
    return ret;
  }
  

  void print(){
    sync();
    for(auto x : data)
      printf("%s\n",std::to_string(x).c_str());
    // printf("Time taken in uploads: %f\n",memoryManager.uploadTime);
    // printf("Average upload time: %f\n",memoryManager.uploadTime/memoryManager.uploadCount);
  }

  std::vector<double> get_data(){
    return data;
  }

  std::vector<double> get_gpu_data(){
    std::vector<double> ret;
    memoryManager.downloadData(id,ret);
    return ret;
  }

  void sync(){ // Download data from GPU to CPU
    ensureSafeAccess();
    memoryManager.downloadData(id,data);
  }

  void upload(){
    ensureSafeAccess();
    memoryManager.uploadData(id,data);
  }

  std::vector<size_t> get_dims(){
    return dims;
  }
  Tensor(const Tensor& other) : data(other.data), dims(other.dims), gpuExecutor(other.gpuExecutor), id(other.id), onlyGPU(other.onlyGPU), transposed(other.transposed) {
    memoryManager.addReference(id);
  }
  ~Tensor(){
    memoryManager.removeReference(id);
    
  }

};
