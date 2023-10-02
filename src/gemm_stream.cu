#include "gemm.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
__global__ void print_matrix(double* A, int M, int N, int istream)
{
    printf("matrix C in stream %d: \n", istream);
    for (int r = 0; r < M; r++)
    {
        for (int c = 0;c < N;c++)
        {
            printf("%f ", A[c * M + r]);
        }
        printf("\n");
    }
}

template<>
void CudaGemm<double>::gemmblas_stream(std::vector<CaseDense<double>>ABC, int M, int N, int K, double alpha, double beta)
{
    // create cuda stream
    cudaStream_t* streams = new cudaStream_t[ABC.size()];
    for (int i = 0; i < ABC.size(); i++)
    {
        cudaStreamCreate(&streams[i]);
    }
    // device memory
    double** d_As = new double* [ABC.size()];
    double** d_Bs = new double* [ABC.size()];
    double** d_Cs = new double* [ABC.size()];

    cublasHandle_t handle;
    cublasCreate(&handle);

    auto start = std::chrono::system_clock::now();
    for (int i = 0;i < ABC.size();i++)
    {
        // auto start_i = std::chrono::system_clock::now();

        // why cudaMallocAsync is slower than cudaMalloc?
        // cudaMallocAsync((void**)&d_As[i], M * K * sizeof(double), streams[i]);
        // cudaMallocAsync((void**)&d_Bs[i], K * N * sizeof(double), streams[i]);
        // cudaMallocAsync((void**)&d_Cs[i], M * N * sizeof(double), streams[i]);
        cudaMalloc((void**)&d_As[i], M * K * sizeof(double));
        cudaMalloc((void**)&d_Bs[i], K * N * sizeof(double));
        cudaMalloc((void**)&d_Cs[i], M * N * sizeof(double));
        // cudaMemcpy(d_As[i], ABC[i].A.data(), M * K * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpyAsync(d_Bs[i], ABC[i].B.data(), K * N * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
        
        cublasSetStream(handle, streams[i]);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_As[i], M, d_Bs[i], K, &beta, d_Cs[i], M);

        // print_matrix<<<1, 1, 0, streams[i]>>>(d_Cs[i], M, N, i);

        // cudaMemcpy(ABC[i].C.data(), d_Cs[i], M * N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(ABC[i].C.data(), d_Cs[i], M * N * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);

        // auto end_i = std::chrono::system_clock::now();
        // std::chrono::duration<double> duration_i = end_i - start_i;
        // std::cout << "stream:" << i << "time elapsed: " << double(duration_i.count()) << " ms" << std::endl;

        // cudaFreeAsync(d_As[i], streams[i]);
        // cudaFreeAsync(d_Bs[i], streams[i]);
        // cudaFreeAsync(d_Cs[i], streams[i]);
        cudaFree(d_As[i]);
        cudaFree(d_Bs[i]);
        cudaFree(d_Cs[i]);
    }

    cublasDestroy(handle);
    for (int i = 0; i < ABC.size(); i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "total time elapsed(GPU): " << double(duration.count()) << " ms" << std::endl;

    delete[] streams;
    delete[] d_As;
    delete[] d_Bs;
    delete[] d_Cs;
}


template<>
void CudaGemm<double>::gemmblas_cpu_ref(std::vector<CaseDense<double>>ABC, int M, int N, int K, double alpha, double beta)
{
    auto start = std::chrono::system_clock::now();
    for (int i = 0;i < ABC.size();i++)
        dgemm_("N", "N", &M, &N, &K, &alpha, ABC[i].A.data(), &M, ABC[i].B.data(), &K, &beta, ABC[i].C.data(), &M);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "total time elapsed(CPU): " << double(duration.count()) << " ms" << std::endl;

}