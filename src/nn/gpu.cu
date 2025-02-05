#include "gpu.h"
#include "memory_manager.h"
#define TILE_SIZE 32
__global__ void negate( const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        b[i] = -a[i];
    }
}

__global__ void reciprocal( const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double epsilon = 1e-8;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        b[i] = 1.0 / (a[i] + epsilon);
    }
}

__global__ void add(const double* a, const double* b, double* c, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] + b[i];
    }
}

__global__ void subtract(const double* a, const double* b, double* c, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] - b[i];
    }
}

__global__ void elementwise_multiply(const double* a, const double* b, double* c, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] * b[i];
    }
}

__global__ void multiply(const double* a, const double b, double* c, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] * b;
    }
}

__global__ void power(const double* a, const double b, double* c, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = pow(a[i], b);
    }
}

__global__ void relu(const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        b[i] = fmax(0.0, a[i]);
    }
}

__global__ void binarilize(const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        b[i] = (a[i] > 0.0) ? 1.0 : 0.0;
    }
}

__global__ void exp(const double* a, double* c, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = exp(a[i]);
    }
}

__global__ void transpose3d(const double* a, double* b, int width, int height, int depth) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

    if (xIndex < width && yIndex < height && zIndex < depth) {
        b[zIndex * width * height + yIndex * width + xIndex] = a[zIndex * width * height + xIndex * height + yIndex];
    }
}

__global__ void matmul(const double* A, const double* B, double* C, int M, int N, int K) {
    //Multiplying matrix A (depthxMxK) with matrix B (KxN) and storing the result in C (depthxMxN)
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int depth = blockIdx.z;
    double sum = 0;

    for (int i = 0; i < (K + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (i * TILE_SIZE + threadIdx.x < K && row < M) {
            tileA[threadIdx.y][threadIdx.x] = A[depth * M * K + row * K + i * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (i * TILE_SIZE + threadIdx.y < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[depth * M * N + row * N + col] = sum;
    }
}

__global__ void sigmoid(const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        b[i] = 1.0 / (1.0 + exp(-a[i]));
    }
}

__global__ void reluDerivative(const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        b[i] = (a[i] > 0.0) ? 1.0 : 0.0;
    }
}

__global__ void sigmoidDerivative(const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sigmoidValue;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sigmoidValue = 1.0 / (1.0 + exp(-a[i]));
        b[i] = sigmoidValue * (1.0 - sigmoidValue);
    }
}

__global__ void softmaxDerivative(const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += exp(a[i]);
    }
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        b[i] = exp(a[i]) / (sum * sum);
    }
}

__global__ void logarithm(const double* a, double* b, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double epsilon = 1e-8;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        b[i] = log(a[i] + epsilon);
    }
}

__global__ void divide(const double* a, const double* b, double* c, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double epsilon = 1e-8;
    for(int i = idx; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] / (b[i] + epsilon);
    }
}

void GPUExecutor::launchNegate(int idA, int idB) { //Negate tensor A and store in B
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = (n + blockSize - 1) / blockSize; // Calculate number of blocks needed
    if(synchronous) {
        negate<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        negate<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchReciprocal(int idA, int idB) { //Calculate reciprocal of tensor A and store in B
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = (n + blockSize - 1) / blockSize; // Calculate number of blocks needed
    if(synchronous) {
        reciprocal<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        reciprocal<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchAdd(int idA, int idB,int idC) { //Add tensor A and B and store in C
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    double *d_dataC = memoryManager.getGpuData(idC);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        add<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, d_dataC, n);
        cudaDeviceSynchronize();
    } else {
        add<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, d_dataC, n);
    }
}

void GPUExecutor::launchSubtract(int idA, int idB,int idC) { //Subtract tensor B from A and store in C
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    double *d_dataC = memoryManager.getGpuData(idC);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        subtract<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, d_dataC, n);
        cudaDeviceSynchronize();
    } else {
        subtract<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, d_dataC, n);
    }
}

void GPUExecutor::launchElementwiseMultiply(int idA, int idB,int idC) { //Multiply tensor A and B elementwise and store in C
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    double *d_dataC = memoryManager.getGpuData(idC);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        elementwise_multiply<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, d_dataC, n);
        cudaDeviceSynchronize();
    } else {
        elementwise_multiply<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, d_dataC, n);
    }
}

void GPUExecutor::launchMultiply(int idA, double scalar, int idB) { //Multiply tensor A by scalar and store in B
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        multiply<<<numBlocks, blockSize, 0>>>(d_dataA, scalar, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        multiply<<<numBlocks, blockSize, 0, stream>>>(d_dataA, scalar, d_dataB, n);
    }
}

void GPUExecutor::launchPower(int idA, double exponent, int idB) { //Raise tensor A to the power of exponent and store in B
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        power<<<numBlocks, blockSize, 0>>>(d_dataA, exponent, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        power<<<numBlocks, blockSize, 0, stream>>>(d_dataA, exponent, d_dataB, n);
    }
}

void GPUExecutor::launchRelu(int idA, int idB) { //Apply ReLU function to tensor A and store in B
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        relu<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        relu<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchBinarilize(int idA, int idB) { //Binarilize tensor A and store in B
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        binarilize<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        binarilize<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchExp(int idA, int idB) { //Calculate exponential of tensor A and store in B
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        exp<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        exp<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchTranspose2d(int idA, int idB,int width,int height) {
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockWidth = 16;
    dim3 blockSize(blockWidth, blockWidth);
    dim3 gridSize((width + blockWidth - 1) / blockWidth, (height + blockWidth - 1) / blockWidth);
    if(synchronous) {
        transpose3d<<<gridSize, blockSize, 0>>>(d_dataA, d_dataB, width, height,1);
        cudaDeviceSynchronize();
    } else {
        transpose3d<<<gridSize, blockSize, 0, stream>>>(d_dataA, d_dataB, width, height,1);
    }
}

void GPUExecutor::launchTranspose3d(int idA, int idB,int width, int height, int depth) {
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockWidth = 8;
    dim3 blockSize(blockWidth, blockWidth, blockWidth);
    dim3 gridSize((width + blockWidth - 1) / blockWidth, (height + blockWidth - 1) / blockWidth, (depth + blockWidth - 1) / blockWidth);
    if(synchronous) {
        transpose3d<<<gridSize, blockSize, 0>>>(d_dataA, d_dataB, width, height, depth);
        cudaDeviceSynchronize();
    } else {
        transpose3d<<<gridSize, blockSize, 0, stream>>>(d_dataA, d_dataB, width, height, depth);
    }
}

void GPUExecutor::launchMatmul(int idA, int idB, int idC,int M,int N,int K, int depth){
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    double *d_dataC = memoryManager.getGpuData(idC);
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, depth);
    int sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(double);
    if(synchronous) {
        matmul<<<gridSize, blockSize, sharedMemSize>>>(d_dataA, d_dataB, d_dataC, M, N, K);
        cudaDeviceSynchronize();
    } else {
        matmul<<<gridSize, blockSize, sharedMemSize, stream>>>(d_dataA, d_dataB, d_dataC, M, N, K);
    }
}

void GPUExecutor::launchSigmoid(int idA,int idB){
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        sigmoid<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        sigmoid<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchReluDerivative(int idA, int idB) {
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        reluDerivative<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        reluDerivative<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchSigmoidDerivative(int idA, int idB) {
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        sigmoidDerivative<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        sigmoidDerivative<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchSoftmaxDerivative(int idA, int idB) {
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        softmaxDerivative<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        softmaxDerivative<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchLog(int idA, int idB) {
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        logarithm<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, n);
        cudaDeviceSynchronize();
    } else {
        logarithm<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, n);
    }
}

void GPUExecutor::launchDivide(int idA, int idB, int idC) {
    cudaSetDevice(deviceId);
    double *d_dataA = memoryManager.getGpuData(idA);
    double *d_dataB = memoryManager.getGpuData(idB);
    double *d_dataC = memoryManager.getGpuData(idC);
    int blockSize = 128;
    int n = memoryManager.getTensorSize(idA);
    int numBlocks = 8;
    if(synchronous) {
        divide<<<numBlocks, blockSize, 0>>>(d_dataA, d_dataB, d_dataC, n);
        cudaDeviceSynchronize();
    } else {
        divide<<<numBlocks, blockSize, 0, stream>>>(d_dataA, d_dataB, d_dataC, n);
    }
}