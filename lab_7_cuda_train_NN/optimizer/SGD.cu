#include "SGD.h"

#include <iostream>

SGD::SGD(std::vector<std::vector<Tensor*>> params, std::vector<std::vector<Tensor*>> grads, float lr)
{
    cublasCreate(&(handle));

    this->params = params;
    this->grads = grads;
    this->lr = lr;
}

SGD::~SGD()
{
    cublasDestroy(this->handle);
}

void SGD::step()
{
    float alpha = -lr;
    for (int i = 0; i < params.size(); ++i)
    {
        for (int p_idx = 0; p_idx < params.at(i).size(); ++p_idx)
        {
            int size = params.at(i).at(p_idx)->height * params.at(i).at(p_idx)->width;
            cublasSaxpy_v2(handle, size, &alpha, grads.at(i).at(p_idx)->tensor, 1, params.at(i).at(p_idx)->tensor, 1);
        }
    }
}

void SGD::zero_grad()
{
    float zero = 0;
    for (int i = 0; i < this->grads.size(); ++i)
    {
        for (auto grad : grads.at(i))
        {
            cublasSscal_v2(handle, grad->height * grad->width, &zero, grad->tensor, 1);
        }
    }
}
