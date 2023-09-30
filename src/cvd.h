#pragma once
#include "vec3_order.h"
#include <map>
#include <vector>
class CVD
{
public:
    CVD() {};
    ~CVD() {};
    void read_CVD(std::string cfile, std::string vfile, std::string dfile);
    std::map<int, std::map<std::pair<int, Vector3_Order<int>>, std::vector<double>>> Cs;
    std::map<int, std::map<std::pair<int, Vector3_Order<int>>, std::vector<double>>> Vs;
    std::map<int, std::map<std::pair<int, Vector3_Order<int>>, std::vector<double>>> Ds;
    //atoms
    int na = 2;
    int nb = 2;
};