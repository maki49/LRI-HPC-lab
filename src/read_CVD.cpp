#include "cvd.h"
#include <fstream>
#include <assert.h>
#include <iostream>

inline std::map<int, std::map<std::pair<int, Vector3_Order<int>>, std::vector<double>>> read_map(std::string file, int na, int nb, int ndim)
{
    std::map<int, std::map<std::pair<int, Vector3_Order<int>>, std::vector<double>>> res;
    std::ifstream ifs(file);
    if (!ifs.is_open())
    {
        std::cout << "Error opening file " << file << std::endl;
        exit(1);
    }
    std::string tmp;
    int ta_read;
    int tb_read;
    std::vector<int> dim(ndim);
    ifs >> tmp;
    while (ifs >> ta_read&& ta_read < na)
    {
        assert(tmp == "ta=");
        std::cout << "ta_read=" << ta_read << std::endl;
        while (ifs >> tmp && tmp == "tb=")
        {
            ifs >> tb_read;
            std::cout << tmp << " " << tb_read << std::endl;
            int i, j, k;
            ifs >> tmp >> i >> j >> k;   // R 

            std::cout << tmp << " " << i << " " << j << " " << k << std::endl;
            ifs >> tmp; // "size="
            int totdim = 1;
            for (int l = 0;l < ndim;++l) {
                ifs >> dim[l];
                totdim *= dim[l];
            }
            std::vector<double> v(totdim);
            for (int i = 0;i < v.size();++i) ifs >> v[i];
            res[ta_read][std::make_pair(tb_read, Vector3_Order<int>(i, j, k))] = std::move(v);
        }
    }
    ifs.close();
    return res;
}
void CVD::read_CVD(std::string cfile, std::string vfile, std::string dfile)
{
    this->Cs = read_map(cfile, this->na, this->nb, 3);
    this->Vs = read_map(vfile, this->na, this->nb, 2);
    this->Ds = read_map(dfile, this->na, this->nb, 2);
}