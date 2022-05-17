#ifndef LINEAR_H
#define LINEAR_H

#include <vector>
#include <cublas_v2.h>
#include "BaseLayer.h"

class Linear: public BaseLayer
{
public:
    Linear(int in_features, int out_features, bool bias);
    ~Linear();

    Tensor* forward(Tensor *input);
    std::vector<Tensor*> get_state();

private:
    Tensor *weights;
    Tensor *bias;
    Tensor *output;
    int in_features;
    int out_features;

    cublasHandle_t handle;
};

#endif