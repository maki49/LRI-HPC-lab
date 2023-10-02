#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>

static const char* _cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
        return "Unknown";
    }
}


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

void cublasAssert(cublasStatus_t code, const char* file, int line, bool abort = true) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Assert: %s %s %d\n", _cublasGetErrorEnum(code), file, line);
        if (abort) exit(code);
    }
}

void cusparseAssert(cusparseStatus_t code, const char* file, int line, bool abort = true)
{
    if (code != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUSPARSE Assert: %s %s %d\n", _cusparseGetErrorEnum(code), file, line);
        if (abort) exit(code);
    }
}

template<typename T>
void alloc_h2d(const T* h, T*& d, const int size)
{
    cudaMalloc((void**)&d, size * sizeof(T));
    cudaMemcpy(d, h, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void alloc_d2h(const T* d, T*& h, const int size)
{
    h = new T[size];
    cudaMemcpy(h, d, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void d2h(const T* d, T* h, const int size)
{
    cudaMemcpy(h, d, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void ad2h(const T* d, T* h, const int size, cudaStream_t stream)
{
    cudaMemcpyAsync(h, d, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
}

template<typename T>
void h2d(const T* h, T* d, const int size)
{
    cudaMemcpy(d, h, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void ah2d(const T* h, T* d, const int size, cudaStream_t stream)
{
    cudaMemcpyAsync(d, h, size * sizeof(T), cudaMemcpyHostToDevice, stream);
}

template<typename T>
void dfree(T* d)
{
    cudaFree(d);
}


//int
template void alloc_h2d<int>(const int* h, int*& d, const int size);
template void alloc_d2h<int>(const int* d, int*& h, const int size);
template void d2h<int>(const int* d, int* h, const int size);
template void ad2h<int>(const int* d, int* h, const int size, cudaStream_t stream);
template void h2d<int>(const int* h, int* d, const int size);
template void ah2d<int>(const int* h, int* d, const int size, cudaStream_t stream);
template void dfree<int>(int* d);

template void alloc_h2d<unsigned long>(const unsigned long* h, unsigned long*& d, const int size);
template void alloc_d2h<unsigned long>(const unsigned long* d, unsigned long*& h, const int size);
template void d2h<unsigned long>(const unsigned long* d, unsigned long* h, const int size);
template void ad2h<unsigned long>(const unsigned long* d, unsigned long* h, const int size, cudaStream_t stream);
template void h2d<unsigned long>(const unsigned long* h, unsigned long* d, const int size);
template void ah2d<unsigned long>(const unsigned long* h, unsigned long* d, const int size, cudaStream_t stream);
template void dfree<unsigned long>(unsigned long* d);


//double
template void alloc_h2d<double>(const double* h, double*& d, const int size);
template void alloc_d2h<double>(const double* d, double*& h, const int size);
template void d2h<double>(const double* d, double* h, const int size);
template void ad2h<double>(const double* d, double* h, const int size, cudaStream_t stream);
template void h2d<double>(const double* h, double* d, const int size);
template void ah2d<double>(const double* h, double* d, const int size, cudaStream_t stream);
template void dfree<double>(double* d);