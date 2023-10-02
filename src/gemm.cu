#include "gemm.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iostream>
#include <assert.h>
#include "utils.h"

template<typename T>
__device__ T get_element(const T* A, int r, int c, int lda)
{
    return A[c * lda + r];
}
template<typename T>
__device__ void set_element(T* A, int r, int c, int lda, const T val)
{
    A[c * lda + r] = val;
}

template<typename T>
T get_element_h(const T* A, int r, int c, int lda)
{
    return A[c * lda + r];
}

template<typename T>
void set_element_h(T* A, int r, int c, int lda, const T val)
{
    A[c * lda + r] = val;
}

template<typename T>
void set_zeros(T* A, const int size)
{
    cudaMemset(A, 0, size * sizeof(T));
}

__global__ void print_device_array(const int* A, const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%d ", A[i]);
    }
    printf("\n");
}
__global__ void print_device_array(const unsigned long* A, const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%lu ", A[i]);
    }
    printf("\n");
}
__global__ void print_device_array(const double* A, const int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("%f ", A[i]);
    }
    printf("\n");
}

__global__ void gemm2d_kernel(double* A, double* B, double* C, int M, int N, int K, double alpha, double beta)
{
    // 2d-structure C(M, N)
    // each thread computes one element of C
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N)
    {
        double sum = 0;
        for (int i = 0; i < K; i++)
        {
            sum += get_element(A, row, i, M) * get_element(B, i, col, K);
        }
        set_element(C, row, col, M, alpha * sum + beta * get_element(C, row, col, M));
    }
}

template<>
void CudaGemm<double>::gemm2d(double* A, double* B, double* C, int M, int N, int K, double alpha, double beta)
{
    int bx = 32;
    int by = 32;
    int gx = (M - 1) / bx + 1;
    int gy = (N - 1) / by + 1;
    dim3 block(bx, by);
    dim3 grid(gx, gy);

    // allocate device memory
    double* dA;
    double* dB;
    double* dC;
    alloc_h2d(A, dA, M * K);
    alloc_h2d(B, dB, K * N);
    alloc_h2d(C, dC, M * N);
    gemm2d_kernel << <grid, block >> > (dA, dB, dC, M, N, K, alpha, beta);
    d2h(dC, C, M * N);
    dfree(dA);
    dfree(dB);
    dfree(dC);
}

template<>
void CudaGemm<double>::gemmblas(double* A, double* B, double* C, int M, int N, int K, double alpha, double beta)
{
    // allocate device memory
    double* dA;
    double* dB;
    double* dC;
    alloc_h2d(A, dA, M * K);
    alloc_h2d(B, dB, K * N);
    alloc_h2d(C, dC, M * N);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, dA, M, dB, K, &beta, dC, M);
    cublasDestroy(handle);

    d2h(dC, C, M * N);
    dfree(dA);
    dfree(dB);
    dfree(dC);
}

template<typename T>
int CudaGemm<T>::dense2csr(const T* A, const int M, const int N, T* V, int* CI, int* RI, const T thr)
{
    int nnz = 0;
    for (int i = 0; i < M; i++)
    {
        int row_nnz = 0;
        for (int j = 0; j < N; j++)
        {
            T v = get_element_h(A, i, j, M);
            if (v > thr)
            {
                V[nnz] = v;
                CI[nnz] = j;
                nnz++;
                row_nnz++;
            }
        }
        RI[i + 1] = nnz;
        assert(RI[i + 1] == RI[i] + row_nnz);
    }
    return nnz;
}

template<typename T>
int CudaGemm<T>::dense2csr(const T* A, const int M, const int N, std::vector<T>& V, std::vector<int>& CI, std::vector<int>& RI, const T thr)
{
    RI.resize(M + 1);
    int nnz = 0;
    for (int i = 0; i < M; i++)
    {
        int row_nnz = 0;
        for (int j = 0; j < N; j++)
        {
            T v = get_element_h(A, i, j, M);
            if (v > thr)
            {
                V.push_back(v);
                CI.push_back(j);
                nnz++;
                row_nnz++;
            }
        }
        RI[i + 1] = nnz;
        assert(RI[i + 1] == RI[i] + row_nnz);
    }
    assert(nnz==V.size() && nnz==CI.size());
    return nnz;
}

template<typename T>
void CudaGemm<T>::csr2dense(const T* V, const int* CI, const int* RI, const int M, const int N, T* A)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = RI[i]; j < RI[i + 1]; j++)
        {
            set_element_h(A, i, CI[j], M, V[j]);
        }
    }
}

template<>
int CudaGemm<double>::cusparsegemm_dmem(const int M, const int N, const int K, const double alpha, const double beta, const int nnzA, int* dRowPtrA, int* dColIndA, double* dValA,
    const int nnzB, int* dRowPtrB, int* dColIndB, double* dValB, int* dRowPtrC, int* dColIndC, double* dValC)
{
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseIndexType_t i32 = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t b0 = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType_t r64f = CUDA_R_64F;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;

    // // test： print A, B
    // std::cout << "A: " << std::endl;
    // std::cout << "nnzA: " << nnzA << std::endl;
    std::cout << "dRowPtrA: " << std::endl;
    print_device_array << <1, 1 >> > (dRowPtrA, M + 1);
    // std::cout << "dColIndA: " << std::endl;
    // print_device_array << <1, 1 >> > (dColIndA, nnzA);
    // std::cout << "dValA: " << std::endl;
    // print_device_array << <1, 1 >> > (dValA, nnzA);
    // std::cout << "B: " << std::endl;
    // std::cout << "nnzB: " << nnzB << std::endl;
    // std::cout << "dRowPtrB: " << std::endl;
    // print_device_array << <1, 1 >> > (dRowPtrB, K + 1);
    // std::cout << "dColIndB: " << std::endl;
    // print_device_array << <1, 1 >> > (dColIndB, nnzB);
    // std::cout << "dValB: " << std::endl;
    // print_device_array << <1, 1 >> > (dValB, nnzB);

    cusparseHandle_t handle;
    cusparseErrcheck(cusparseCreate(&handle));
    cusparseSpMatDescr_t matA, matB, matC;
    cusparseErrcheck(cusparseCreateCsr(&matA, M, K, nnzA, dRowPtrA, dColIndA, dValA,
        i32, i32, b0, r64f));
    cusparseErrcheck(cusparseCreateCsr(&matB, K, N, nnzB, dRowPtrB, dColIndB, dValB,
        i32, i32, b0, r64f));
    cusparseErrcheck(cusparseCreateCsr(&matC, M, N, 0, dRowPtrC, NULL, NULL,
        i32, i32, b0, r64f));

    // cannot pass a NULL-dRowPtrC to cusparseCreateCsr
    // cusparseCreateCsr(&matC, M, N, 0, NULL, NULL, NULL,
    //     i32, i32, b0, r64f);

    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseErrcheck(cusparseSpGEMM_createDescr(&spgemmDesc));
    // ask buffersize bytes for external memory
    void* dbuffer1 = NULL, * dbuffer2 = NULL;
    size_t buffersize1 = 0, buffersize2 = 0;
    cusparseErrcheck(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc, &buffersize1, NULL));
    cudaErrcheck(cudaMalloc(&dbuffer1, buffersize1));

    // inspect the matrices A and B to understand the memory requiremnets for the next step
    ///if use unsigned long as index,  `CUSPARSE Assert: <unknown> here` will occur
    cusparseErrcheck(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc, &buffersize1, dbuffer1));
    // ask buffersize2 bytes for external memory
    cusparseErrcheck(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc, &buffersize2, NULL));
    cudaErrcheck(cudaMalloc(&dbuffer2, buffersize2));
    //compute A*B
    cusparseErrcheck(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc, &buffersize2, dbuffer2));
    // get C's non-zero elements: nnzC1

    cudaErrcheck(cudaFree(dbuffer1));
    cudaErrcheck(cudaFree(dbuffer2));

    // copy C back to host
    int64_t nrowC, ncolC, nnzC;
    cusparseErrcheck(cusparseSpMatGetSize(matC, &nrowC, &ncolC, &nnzC));

    // allocate device memory for C using nnzC
    // seems I can't allocate device memory in subfunction and access it in main function
    // which will cause segmentation fault
    // if (if_Malloc_C_Col_Val)
    // {
    //     cudaErrcheck(cudaMalloc((void**)&dColIndC, nnzC * sizeof(int)));
    //     cudaErrcheck(cudaMalloc((void**)&dValC, nnzC * sizeof(double)));
    // }

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update C with the new pointer
    cusparseErrcheck(cusparseCsrSetPointers(matC, dRowPtrC, dColIndC, dValC));

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to C
    cusparseErrcheck(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc));

    // destroy matrix/vector descriptors
    cusparseErrcheck(cusparseSpGEMM_destroyDescr(spgemmDesc));
    cusparseErrcheck(cusparseDestroySpMat(matA));
    cusparseErrcheck(cusparseDestroySpMat(matB));
    cusparseErrcheck(cusparseDestroySpMat(matC));
    cusparseErrcheck(cusparseDestroy(handle));

    // std::cout << "C: " << std::endl;
    // std::cout << "nnzC: " << nnzC << std::endl;
    // print_device_array << <1, 1 >> > (dValC, nnzC);

    return nnzC;
}
template<>
void CudaGemm<double>::gemmsparse_csr(double* A, double* B, double* C, int M, int N, int K, double alpha, double beta)
{

    // reference: https://docs.nvidia.com/cuda/cusparse/#cusparsespgemm
    // =============host==============

    std::vector<int> hRowPtrA(M + 1);
    std::vector<int> hColIndA;
    std::vector<double> hValA;
    std::vector<int> hRowPtrB(K + 1);
    std::vector<int> hColIndB;
    std::vector<double> hValB;
    std::vector<int> hRowPtrC(M + 1);
    std::vector<int> hColIndC;
    std::vector<double> hValC;


    // convert dense to CSR
    int nnzA = dense2csr(A, M, K, hValA, hColIndA, hRowPtrA);
    int nnzB = dense2csr(B, K, N, hValB, hColIndB, hRowPtrB);

    // =============device==============
    int* dRowPtrA;
    int* dColIndA;
    double* dValA;
    int* dRowPtrB;
    int* dColIndB;
    double* dValB;
    int* dRowPtrC;
    int* dColIndC;
    double* dValC;


    alloc_h2d(hRowPtrA.data(), dRowPtrA, M + 1);
    alloc_h2d(hColIndA.data(), dColIndA, nnzA);
    alloc_h2d(hValA.data(), dValA, nnzA);
    alloc_h2d(hRowPtrB.data(), dRowPtrB, K + 1);
    alloc_h2d(hColIndB.data(), dColIndB, nnzB);
    alloc_h2d(hValB.data(), dValB, nnzB);
    alloc_h2d(hRowPtrC.data(), dRowPtrC, M + 1);
    // alloc_h2d(hColIndC.data(), dColIndC, 0);
    // alloc_h2d(hValC.data(), dValC, 0);

    int maxnnzC = M * N;
    // allocate device memory for C using maxnnzC
    cudaErrcheck(cudaMalloc((void**)&dColIndC, maxnnzC * sizeof(int)));
    cudaErrcheck(cudaMalloc((void**)&dValC, maxnnzC * sizeof(double)));

    int nnzC = cusparsegemm_dmem(M, N, K, alpha, beta, nnzA, dRowPtrA, dColIndA, dValA, nnzB, dRowPtrB, dColIndB, dValB, dRowPtrC, dColIndC, dValC);

    // print device memory
    // must use kernel function (executed on device) to print device memory
    print_device_array << <1, 1 >> > (dValC, nnzC);

    
    // copy C (CSR) from device to host
    hColIndC.resize(nnzC);
    hValC.resize(nnzC);
    d2h(dRowPtrC, hRowPtrC.data(), M + 1);
    d2h(dColIndC, hColIndC.data(), nnzC);
    d2h(dValC, hValC.data(), nnzC);

    // free device memory
    cudaFree(dRowPtrA);
    cudaFree(dColIndA);
    cudaFree(dValA);
    cudaFree(dRowPtrB);
    cudaFree(dColIndB);
    cudaFree(dValB);
    cudaFree(dRowPtrC);
    cudaFree(dColIndC);
    cudaFree(dValC);
    

    // convert C (CSR) to dense
    csr2dense(hValC.data(), hColIndC.data(), hRowPtrC.data(), M, N, C);
    
}
template class CudaGemm<double>;