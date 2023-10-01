#pragma once
#include <map>
#include <vector>
#include <RI/physics/Exx.h>
class RI_Tensor
{
public:
    RI_Tensor() {};
    ~RI_Tensor() {};
    void read_CVD(std::string cfile, std::string vfile, std::string dfile);
    void cal_Hexxs_lri_cpu();

private:
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> Cs;
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> Vs;
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> Ds;
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> Hs;
    //atoms
    int na = 2;
    int nb = 2;
};