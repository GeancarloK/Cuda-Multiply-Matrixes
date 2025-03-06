#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <curand_kernel.h>  
#define SIZE 2000

__global__ void setA(int *a, unsigned long long seed){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, index, 0, &state);
    a[index] = curand(&state) % 101;
}

__global__ void setB(int *b, unsigned long long seed){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init(seed, index, 0, &state);
    b[index] = curand(&state) % 101;
}

__global__ void multM(int *a, int *b, int *c){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = 0; i < SIZE; i++)
    {
        int index_a = i + blockIdx.x * blockDim.x;
        int index_b = threadIdx.x + i * blockDim.x;
        c[index] += a[index_a] * b[index_b];
    }
}

void printM(int *m){
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            std::cout << m[i*SIZE+j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int numThreads = SIZE;
    int numBlocks = SIZE;
    int tam = SIZE * SIZE * sizeof(int);
    int *a = (int*)malloc(tam);
    int *aGPU;
    int *b = (int*)malloc(tam);
    int *bGPU;
    int *c = (int*)malloc(tam);
    int *cGPU;
    
    cudaMalloc((void**)&aGPU, tam);
    cudaMalloc((void**)&bGPU, tam);
    cudaMalloc((void**)&cGPU, tam);

    unsigned long long seed = time(NULL); // Semente criada na CPU

    setA<<<numBlocks, numThreads>>>(aGPU, seed);
    cudaDeviceSynchronize();
    
    setB<<<numBlocks, numThreads>>>(bGPU, seed + 1);
    cudaDeviceSynchronize();
    
    cudaMemcpy(a, aGPU, tam, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, bGPU, tam, cudaMemcpyDeviceToHost);

    //printM(a);
    //printM(b);

    auto start = std::chrono::high_resolution_clock::now();
    cudaMemset(cGPU, 0, tam);
    multM<<<numBlocks, numThreads>>>(aGPU, bGPU, cGPU);
    cudaDeviceSynchronize();
    cudaMemcpy(c, cGPU, tam, cudaMemcpyDeviceToHost);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Tempo de execução GPU: " << duration.count() << " segundos" << std::endl;
    //printM(c);

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < SIZE; i++)
    {
        for(int j = 0; j < SIZE; j++)
        {
            c[i*SIZE+j] = 0;
            for(int k = 0; k < SIZE; k++)
            {
                c[i*SIZE+j] += a[i*SIZE+k] * b[k*SIZE+j];
            }
        }
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end - start;
    std::cout << "Tempo de execução CPU: " << duration2.count() << " segundos" << std::endl;
    std::cout << "CPU eh " << duration2/duration << " vezes mais lenta q GPU para mult matriz " << SIZE << " x " << SIZE << std::endl;

    free(a);
    cudaFree(aGPU);
    free(b);
    cudaFree(bGPU);
    free(c);
    cudaFree(cGPU);

    return 0;
}
