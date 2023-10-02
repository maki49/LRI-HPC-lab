#include "utils.h"
#include "gemm.h"
#include <assert.h>
#include <chrono>
#include <iostream>

inline void mats_to_device_csr(std::vector<CaseDense<double>>ABC, const int M, const int N, const int K, const int nmat, const int mat_offset, cudaStream_t stream,
    int* d_RowOffsetA, int* d_ColIdxA, double* d_ValA,
    int* d_RowOffsetB, int* d_ColIdxB, double* d_ValB)
{
    auto set_row_offset = [&nmat, &stream](int* d_RowOffset, int nr, int nc) -> void
        {
            std::vector<int> h_RowOffset(nr * nmat + 1);
            for (int i = 0;i < nr * nmat + 1;++i) h_RowOffset[i] = i * nc;
            ah2d(h_RowOffset.data(), d_RowOffset, h_RowOffset.size(), stream);
        };

    set_row_offset(d_RowOffsetA, M, K);
    set_row_offset(d_RowOffsetB, K, N);

    auto set_col_idx = [&nmat, &stream](int* d_ColIdx, int nr, int nc) ->void
        {
            int nnz = nr * nc * nmat;
            std::vector<int> h_ColIdx(nnz);
            for (int m = 0;m < nmat;++m)
                for (int i = 0;i < nr;++i)
                    for (int j = 0;j < nc;++j)
                        h_ColIdx[m * nr * nc + i * nc + j] = nc * m + j;
            ah2d(h_ColIdx.data(), d_ColIdx, h_ColIdx.size(), stream);
            // test: output h_ColIdx
            // std::cout << "h_ColIdx: " << std::endl;
            // for (int i = 0;i < h_ColIdx.size();++i)
            // {
            //     std::cout << h_ColIdx[i] << " ";
            //     if ((i + 1) % 10 == 0) std::cout << std::endl;
            // }

        };
    set_col_idx(d_ColIdxA, M, K);
    set_col_idx(d_ColIdxB, K, N);

    for (int m = mat_offset;m < mat_offset + nmat;++m)
    {
        ah2d(ABC[m].A.data(), d_ValA + m * M * K, M * K, stream);
        ah2d(ABC[m].B.data(), d_ValB + m * K * N, K * N, stream);
    }
}

template<>
void CudaGemm<double>::gemmsparse_stream_block_csr(std::vector<CaseDense<double>>ABC, int M, int N, int K, double alpha, double beta, int nstream)
{
    std::cout << "begin" << std::endl;
    int nmat_prev = (ABC.size() % nstream) ? ABC.size() / nstream + 1 : ABC.size() / nstream;   // stream 0, 1, ..., nstream-2
    int nmat_last = (ABC.size() % nmat_prev) ? ABC.size() % nmat_prev : nmat_prev;       //the last stream
    std::cout << "nstream=" << nstream << std::endl;
    std::cout << "nmat_prev=" << nmat_prev << std::endl;
    int nmat = nmat_prev;   // current stream


    int* d_RowOffsetA;
    int* d_ColIdxA;
    double* d_ValA;

    int* d_RowOffsetB;
    int* d_ColIdxB;
    double* d_ValB;

    int* d_RowOffsetC;
    int* d_ColIdxC;
    double* d_ValC;

    int nnzA = M * K * nmat;
    int nnzB = K * N * nmat;
    int nnzC = M * N * nmat;

    //allocate device memory for CSR matrices
    cudaErrcheck(cudaMalloc((void**)&d_RowOffsetA, sizeof(int) * (M * nmat + 1) * nstream));
    cudaErrcheck(cudaMalloc((void**)&d_ColIdxA, sizeof(int) * nnzA * nstream));
    cudaErrcheck(cudaMalloc((void**)&d_ValA, sizeof(double) * nnzA * nstream));

    cudaErrcheck(cudaMalloc((void**)&d_RowOffsetB, sizeof(int) * (K * nmat + 1) * nstream));
    cudaErrcheck(cudaMalloc((void**)&d_ColIdxB, sizeof(int) * nnzB * nstream));
    cudaErrcheck(cudaMalloc((void**)&d_ValB, sizeof(double) * nnzB * nstream));

    cudaErrcheck(cudaMalloc((void**)&d_RowOffsetC, sizeof(int) * (M * nmat + 1) * nstream));
    cudaErrcheck(cudaMalloc((void**)&d_ColIdxC, sizeof(int) * nnzC * nstream));
    cudaErrcheck(cudaMalloc((void**)&d_ValC, sizeof(double) * nnzC * nstream));

    // cudaErrcheck(cudaFree(d_RowOffsetA));
    // cudaErrcheck(cudaMalloc((void**)&d_RowOffsetA, sizeof(int) * (M * nmat + 1) * nstream));


    //create streams
    cudaStream_t* streams = new cudaStream_t[nstream];

    for (int i = 0;i < nstream;++i)
    {
        cudaErrcheck(cudaStreamCreate(&streams[i]));
    }

    std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
    auto start = std::chrono::system_clock::now();
    for (int i = 0;i < nstream;++i)
    {
        std::cout << "i=" << i << ", nmat=" << nmat << std::endl;
        std::cout << "nnzA=" << nnzA << std::endl;
        int mat_offset = i * nmat_prev;
        int ARow_offset = (M * nmat_prev + 1) * i;
        int BRow_offset = (K * nmat_prev + 1) * i;
        int CRow_offset = (M * nmat_prev + 1) * i;
        int Aval_offset = M * N * nmat_prev * i;
        int Bval_offset = M * N * nmat_prev * i;
        int Cval_offset = M * N * nmat_prev * i;

        mats_to_device_csr(ABC, M, N, K, nmat, mat_offset, streams[i],
            &d_RowOffsetA[ARow_offset], &d_ColIdxA[Aval_offset], &d_ValA[Aval_offset],
            &d_RowOffsetB[BRow_offset], &d_ColIdxB[Bval_offset], &d_ValB[Bval_offset]);

        //call cusparse
        int nnzC_cal = cusparsegemm_dmem(M * nmat, N * nmat, K * nmat, alpha, beta, nnzA,
            &d_RowOffsetA[ARow_offset], &d_ColIdxA[Aval_offset], &d_ValA[Aval_offset],
            nnzB, &d_RowOffsetB[BRow_offset], &d_ColIdxB[Bval_offset], &d_ValB[Bval_offset],
            &d_RowOffsetC[CRow_offset], &d_ColIdxC[Cval_offset], &d_ValC[Cval_offset]);

        // std::cout << "nnz_cal: " << nnzC_cal << std::endl;
        // std::cout << "nnzC: " << nnzC << std::endl;
        // assert(nnzC_cal == nnzC);

        //copy C back to host
        for (int m = mat_offset; m < mat_offset + nmat;++m)
            cudaMemcpyAsync(d_ValC + m * M * N, ABC[m].C.data(), M * N, cudaMemcpyDeviceToHost, streams[i]);
        if (i == nstream - 2 && nmat_last > 0)
        {
            nmat = nmat_last;
            nnzA = M * K * nmat;
            nnzB = K * N * nmat;
            nnzC = M * N * nmat;
        }
    }

    for (int i = 0; i < nstream; i++)
    {
        cudaErrcheck(cudaStreamSynchronize(streams[i]));
        cudaErrcheck(cudaStreamDestroy(streams[i]));
    }
    //free device memory
    cudaErrcheck(cudaFree(d_RowOffsetA));   ///an illegal memory access was encountered, when the matrix is large (but only (50*50)*(50*50)...)
    cudaErrcheck(cudaFree(d_ColIdxA));
    cudaErrcheck(cudaFree(d_ValA));

    cudaErrcheck(cudaFree(d_RowOffsetB));
    cudaErrcheck(cudaFree(d_ColIdxB));
    cudaErrcheck(cudaFree(d_ValB));

    cudaErrcheck(cudaFree(d_RowOffsetC));
    cudaErrcheck(cudaFree(d_ColIdxC));
    cudaErrcheck(cudaFree(d_ValC));

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "total time elapsed(GPU): " << double(duration.count()) << " ms" << std::endl;

    delete[] streams;
}