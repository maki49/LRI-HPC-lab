#include "cvd.h"
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
int main()
{
    test_gemm();


    
    // CVD cvd;
    // cvd.read_CVD("../files/Cs.txt", "../files/Vs.txt", "../files/Ds.txt");
    return 0;
}