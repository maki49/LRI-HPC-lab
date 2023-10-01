#pragma once
#include <vector>
extern "C"
{
    void dgemm_(const char* const transA, const char* const transB, const int* const m, const int* const n, const int* const k,
        const double* const alpha, const double* const A, const int* const ldA, const double* const B, const int* const ldB,
        const double* const beta, double* const C, const int* const ldC);
}
template <typename T>
struct CaseDense
{
    CaseDense() = default;
    CaseDense(const std::vector<T>& A, const std::vector<T>& B, const std::vector<T>& C)
        : A(A), B(B), C(C) {}
    CaseDense(std::vector<T>&& A, std::vector<T>&& B, std::vector<T>&& C)
        : A(std::move(A)), B(std::move(B)), C(std::move(C)) {}
    std::vector<T> A;
    std::vector<T> B;
    std::vector<T> C;
};

template <typename T>
class CudaGemm
{
public:
    //1. 2d grid and block, for loop
    static void gemm2d(T* A, T* B, T* C, int M, int N, int K, T alpha, T beta);
    static void gemmblas(T* A, T* B, T* C, int M, int N, int K, T alpha, T beta);
    static void gemmsparse_csr(T* A, T* B, T* C, int M, int N, int K, T alpha, T beta);
    static void gemmblas_stream(std::vector<CaseDense<T>>ABC, int M, int N, int K, T alpha, T beta);
    static void gemmblas_cpu_ref(std::vector<CaseDense<T>>ABC, int M, int N, int K, T alpha, T beta);
private:
    static int dense2csr(const T* A, const int M, const int N, T* V, int* CI, int* RI, const T thr = 0);
    static int dense2csr(const T* A, const int M, const int N, std::vector<T>& V, std::vector<int>& CI, std::vector<int>& RI, const T thr = 0);
    static void csr2dense(const T* V, const int* CI, const int* RI, const int M, const int N, T* A);
};