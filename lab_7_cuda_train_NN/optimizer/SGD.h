#ifndef SGD_H
#define SGD_H

#include <cublas_v2.h>
#include <vector>
#include "../Tensor.h"

class SGD
{
public:
    SGD(std::vector<std::vector<Tensor*>> params, std::vector<std::vector<Tensor*>> grads, float lr);
    ~SGD();

    void step();
    void zero_grad();

private:
    std::vector<std::vector<Tensor*>> params;
    std::vector<std::vector<Tensor*>> grads;
    float lr;

    cublasHandle_t handle;
};

#endif