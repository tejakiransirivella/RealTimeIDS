#pragma once
#include <cuda_runtime.h>
#include <iostream>
class GPUExecutor {
public:
    bool synchronous;
    int deviceId;
    cudaStream_t stream;
    GPUExecutor(int deviceId = 0) : synchronous(true), deviceId(deviceId){}
    GPUExecutor(cudaStream_t stream,int deviceId = 0) : synchronous(false), deviceId(deviceId), stream(stream) {
        cudaStreamCreate(&stream);
    }
    void launchNegate(int idA,int idB);
    void launchReciprocal(int idA,int idB);
    void launchAdd(int idA, int idB,int idC);
    void launchSubtract(int idA, int idB,int idC);
    void launchElementwiseMultiply(int idA, int idB,int idC);
    void launchMultiply(int idA, double scalar, int idB);
    void launchPower(int idA, double exponent, int idB);
    void launchRelu(int idA, int idB);
    void launchExp(int idA, int idB);
    void launchBinarilize(int idA, int idB);
    void launchTranspose2d(int idA, int idB,int width,int height);
    void launchTranspose3d(int idA, int idB,int width,int height,int depth);
    void launchMatmul(int idA, int idB, int idC,int M, int N,int K,int depth);
    void launchSigmoid(int idA, int idB);
    void launchSigmoidDerivative(int idA, int idB);
    void launchReluDerivative(int idA, int idB);
    void launchSoftmaxDerivative(int idA, int idB);
    void launchLog(int idA, int idB);
    void launchDivide(int idA, int idB, int idC);
    ~GPUExecutor() {
        if (!synchronous) {
            cudaStreamDestroy(stream);
        }
    }

};