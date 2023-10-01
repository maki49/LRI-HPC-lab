#include "ri_tensor.h"
#include "gemm.h"
#include <fstream>

void test_gemm()
{
    std::vector<double> mA =
    { 1,0,5,0,
    0,4,0,8,
    2,0,6,0,
    3,0,7,9 };
    std::vector<double> mB =
    { 1,0,5,0,
    0,3,6,8,
    0,0,7,0,
    2,4,0,0 };
    std::vector<double> mC =
    { 11,0,35,0,
    36,12,92,96,
    14,0,42,0,
    2,16,10,32 };
    CaseDense cd(mA, mB, mC);

    std::vector<double> mC_res(16, 0);
    double alpha = 1.0;
    double beta = 0.0;

    std::cout << "gemm2d: " << std::endl;
    CudaGemm<double>::gemm2d(mA.data(), mB.data(), mC_res.data(), 4, 4, 4, alpha, beta);
    for (int i = 0; i < 16; i++)
    {
        std::cout << mC_res[i] << " ";
    }
    std::cout << std::endl;

    mC_res.assign(16, 0);
    std::cout << "gemmblas: " << std::endl;
    CudaGemm<double>::gemmblas(mA.data(), mB.data(), mC_res.data(), 4, 4, 4, alpha, beta);
    for (int i = 0; i < 16; i++)
    {
        std::cout << mC_res[i] << " ";
    }
    std::cout << std::endl;

    mC_res.assign(16, 0);
    std::cout << "gemmsparse_csr: " << std::endl;
    CudaGemm<double>::gemmsparse_csr(mA.data(), mB.data(), mC_res.data(), 4, 4, 4, alpha, beta);
    for (int i = 0; i < 16; i++)
    {
        std::cout << mC_res[i] << " ";
    }
    std::cout << std::endl;
}

void test_gemm_stream(int nmat, int m, int n, int k)
{
    std::vector<CaseDense<double>> cases(nmat);
    for(int i = 0; i < nmat; i++)
    {
        cases[i].A.assign(m*k, i);
        cases[i].B.assign(k*n, i);
        cases[i].C.assign(m*n, 0);
    }
    CudaGemm<double>::gemmblas_stream(cases, m, n, k, 1.0, 0.0);
    CudaGemm<double>::gemmblas_cpu_ref(cases, m, n, k, 1.0, 0.0);
}

void test_libri_hs_cpu(int argc, char* argv[])
{
    int mpi_init_provide;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_provide);
    RI_Tensor ri;
    ri.read_CVD("../files/Cs.txt", "../files/Vs.txt", "../files/Ds.txt");
    ri.cal_Hexxs_lri_cpu();
    MPI_Finalize();
}

int main(int argc, char* argv[])
{
    // test_gemm();

    // test gemm stream
    // total time elapsed(GPU): 1.54035 ms
    // total time elapsed(CPU) : 0.160346 ms
    int nmat = 10000;
    int m = 13 * 13, n = 65, k = 65;    //CV
    test_gemm_stream(nmat, m, n, k);
    
    // test_libri_hs_cpu(argc, argv);


    return 0;
}