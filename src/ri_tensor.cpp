#include "ri_tensor.h"
#include <fstream>
#include <assert.h>
#include <iostream>
#include <RI/physics/Exx.h>

inline std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> read_map(std::string file, int na, int nb, int ndim)
{
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> res;
    std::ifstream ifs(file);
    if (!ifs.is_open())
    {
        std::cout << "Error opening file " << file << std::endl;
        exit(1);
    }
    std::string tmp;
    int ta_read;
    int tb_read;
    std::vector<unsigned long> dim(ndim);
    ifs >> tmp;
    while (ifs >> ta_read && ta_read < na)
    {
        assert(tmp == "ta=");
        while (ifs >> tmp && tmp == "tb=")
        {
            ifs >> tb_read;
            int i, j, k;
            ifs >> tmp >> i >> j >> k;   // R 
            ifs >> tmp; // "size="
            unsigned long totdim = 1;
            for (int l = 0;l < ndim;++l) {
                ifs >> dim[l];
                totdim *= dim[l];
            }
            RI::Tensor<double> v({ totdim });
            if (ndim == 3) v = v.reshape({ dim[0], dim[1], dim[2] });
            else v = v.reshape({ dim[0], dim[1] });
            assert(v.shape.size() == ndim);
            assert(totdim == v.get_shape_all());
            for (int i = 0;i < totdim;++i) ifs >> v.ptr()[i];
            res[ta_read][std::make_pair(tb_read, std::array<int, 3>({ i, j, k }))] = std::move(v);
        }
    }
    ifs.close();
    return res;
}
void RI_Tensor::read_CVD(std::string cfile, std::string vfile, std::string dfile)
{
    this->Cs = read_map(cfile, this->na, this->nb, 3);
    this->Vs = read_map(vfile, this->na, this->nb, 2);
    this->Ds = read_map(dfile, this->na, this->nb, 2);
}

inline void compare_tensor_map(std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> m1,
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> m2)
{
    assert(m1.size() == m2.size());
    for (auto it1 = m1.begin();it1 != m1.end();++it1)
    {
        assert(m2.find(it1->first) != m2.end());
        assert(it1->second.size() == m2[it1->first].size());
        for (auto it2 = it1->second.begin();it2 != it1->second.end();++it2)
        {
            assert(m2[it1->first].find(it2->first) != m2[it1->first].end());
            assert(it2->second.shape.size() == m2[it1->first][it2->first].shape.size());
            for (int i = 0;i < it2->second.get_shape_all();++i)
            {
                assert(std::abs(it2->second.ptr()[i] - m2[it1->first][it2->first].ptr()[i]) < 1e-4);
            }
        }
    }
}

void RI_Tensor::cal_Hexxs_lri_cpu()
{
    std::cout << "cal_Hexxs_lri_cpu" << std::endl;
    RI::Exx<int, int, 3, double> exx;
    const std::map<int, std::array<double, 3>> atoms_pos = {
        { 0, {0.677084, 0.677084, 0.677084} },
        { 1, {4.73959, 4.73959, 4.73959} } };
    const std::array<std::array<double, 3>, 3> latvec = {
        {{ 0, 2.70834, 2.70834 },
        { 2.70834, 0, 2.70834 },
        { 2.70834, 2.70834, 0 }} };
    const std::array<int, 3> period = { 2, 3, 4 };  // kv->nmp

    std::cout << "set_parallel" << std::endl;
    exx.set_parallel(MPI_COMM_WORLD, atoms_pos, latvec, period);
    std::cout << "set_thr" << std::endl;
    exx.set_csm_threshold(1e-7);
    std::cout << "set_Cs" << std::endl;
    exx.set_Cs(this->Cs, 1e-4);
    std::cout << "set_Vs" << std::endl;
    exx.set_Vs(this->Vs, 1e-1);
    std::cout << "set_Ds" << std::endl;
    exx.set_Ds(this->Ds, 1e-4);
    std::cout << "cal_Hs" << std::endl;
    exx.cal_Hs();
    this->Hs = read_map("../files/Hs.txt", this->na, this->nb, 2);
    std::cout << "check_Hs" << std::endl;
    compare_tensor_map(this->Hs, exx.Hs);
}