#include "gemm.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <iostream>
#include <assert.h>

#define cusparseErrcheck(err) {cusparseAssert(err, __FILE__, __LINE__);}

static const char* _cusparseGetErrorEnum(cusparseStatus_t error) {
    switch (error) {
    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";

    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";

    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";

    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
}

inline void cusparseAssert(cusparseStatus_t code, const char* file, int line, bool abort = true)
{
    if (code != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUSPARSE Assert: %s %s %d\n", _cusparseGetErrorEnum(code), file, line);
        if (abort) exit(code);
    }
}

inline void alloc_h2d(const int* h, int*& d, const int size)
{
    cudaMalloc((void**)&d, size * sizeof(int));
    cudaMemcpy(d, h, size * sizeof(int), cudaMemcpyHostToDevice);
}
inline void alloc_h2d(const double* h, double*& d, const int size)
{
    cudaMalloc((void**)&d, size * sizeof(double));
    cudaMemcpy(d, h, size * sizeof(double), cudaMemcpyHostToDevice);
}

inline void alloc_d2h(const int* d, int*& h, const int size)
{
    h = new int[size];
    cudaMemcpy(h, d, size * sizeof(int), cudaMemcpyDeviceToHost);
}
inline void alloc_d2h(const double* d, double*& h, const int size)
{
    h = new double[size];
    cudaMemcpy(h, d, size * sizeof(double), cudaMemcpyDeviceToHost);
}

inline void d2h(const int* d, int* h, const int size)
{
    cudaMemcpy(h, d, size * sizeof(int), cudaMemcpyDeviceToHost);
}
inline void d2h(const double* d, double* h, const int size)
{
    cudaMemcpy(h, d, size * sizeof(double), cudaMemcpyDeviceToHost);
}

inline void h2d(const int* h, int* d, const int size)
{
    cudaMemcpy(d, h, size * sizeof(int), cudaMemcpyHostToDevice);
}
inline void h2d(const double* h, double* d, const int size)
{
    cudaMemcpy(d, h, size * sizeof(double), cudaMemcpyHostToDevice);
}

inline void dfree(int* d)
{
    cudaFree(d);
}
inline void dfree(double* d)
{
    cudaFree(d);
}

__device__ double get_element(const double* A, int r, int c, int lda)
{
    return A[c * lda + r];
}

__device__ void set_element(double* A, int r, int c, int lda, const double val)
{
    A[c * lda + r] = val;
}

__host__ double get_element_h(const double* A, int r, int c, int lda)
{
    return A[c * lda + r];
}

__host__ void set_element_h(double* A, int r, int c, int lda, const double val)
{
    A[c * lda + r] = val;
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

template<typename T>
void set_zeros(T* A, const int size)
{
    cudaMemset(A, 0, size * sizeof(T));
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

    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseIndexType_t i32 = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t b0 = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType_t r64f = CUDA_R_64F;
    cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;


    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseSpMatDescr_t matA, matB, matC;
    cusparseCreateCsr(&matA, M, K, nnzA, dRowPtrA, dColIndA, dValA,
        i32, i32, b0, r64f);
    cusparseCreateCsr(&matB, K, N, nnzB, dRowPtrB, dColIndB, dValB,
        i32, i32, b0, r64f);
    cusparseCreateCsr(&matC, M, N, 0, dRowPtrC, NULL, NULL,
        i32, i32, b0, r64f);

    // cannot pass a NULL-dRowPtrC to cusparseCreateCsr
    // cusparseCreateCsr(&matC, M, N, 0, NULL, NULL, NULL,
    //     i32, i32, b0, r64f);

    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseErrcheck(cusparseSpGEMM_createDescr(&spgemmDesc));
    // ask buffersize bytes for external memory
    void* dbuffer1 = NULL, * dbuffer2 = NULL;
    size_t buffersize1 = 0, buffersize2 = 0;
    cusparseErrcheck(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc, &buffersize1, NULL));
    cudaMalloc(&dbuffer1, buffersize1);
    // inspect the matrices A and B to understand the memory requiremnets for the next step
    cusparseErrcheck(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc, &buffersize1, dbuffer1));
    // ask buffersize2 bytes for external memory
    cusparseErrcheck(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc, &buffersize2, NULL));
    cudaMalloc(&dbuffer2, buffersize2);
    //compute A*B
    cusparseErrcheck(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, r64f, alg, spgemmDesc, &buffersize2, dbuffer2));
    // get C's non-zero elements: nnzC1
    
    // copy C back to host
    int64_t nrowC, ncolC, nnzC;
    cusparseErrcheck(cusparseSpMatGetSize(matC, &nrowC, &ncolC, &nnzC));

    // allocate device memory for C
    cudaMalloc((void**)&dColIndC, nnzC * sizeof(int));
    cudaMalloc((void**)&dValC, nnzC * sizeof(double));

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