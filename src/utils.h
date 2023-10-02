#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

#define cudaErrcheck(res) {                                             \
    if (res != cudaSuccess) {                                           \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}
#define cublasErrcheck(res) { cublasAssert((res), __FILE__, __LINE__); }
#define cusparseErrcheck(err) {cusparseAssert(err, __FILE__, __LINE__);}

static const char* _cusparseGetErrorEnum(cusparseStatus_t error);
static const char* _cublasGetErrorEnum(cublasStatus_t error);

void cusparseAssert(cusparseStatus_t code, const char* file, int line, bool abort = true);
void cublasAssert(cublasStatus_t code, const char* file, int line, bool abort = true);

template<typename T>
void alloc_h2d(const T* h, T*& d, const int size);
template<typename T>
void alloc_d2h(const T* d, T*& h, const int size);
template<typename T>
void d2h(const T* d, T* h, const int size);
template<typename T>
void ad2h(const T* d, T* h, const int size, cudaStream_t stream);
template<typename T>
void h2d(const T* h, T* d, const int size);
template<typename T>
void ah2d(const T* h, T* d, const int size, cudaStream_t stream);

template<typename T>
void dfree(T* d);