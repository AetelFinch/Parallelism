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
    Tensor* backward(Tensor *grad);

    std::vector<Tensor*> get_state();
    std::vector<Tensor*> get_grads();

private:
    Tensor *weights;
    Tensor *bias;
    Tensor *input;
    
    Tensor *grad_weights;
    Tensor *grad_bias;
    
    int in_features;
    int out_features;
    bool isBias;

    cublasHandle_t handle;
};

#endif