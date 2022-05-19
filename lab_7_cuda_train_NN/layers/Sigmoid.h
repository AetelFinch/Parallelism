#ifndef SIGMOID_H
#define SIGMOID_H

#include "BaseLayer.h"

class Sigmoid: public BaseLayer
{
public:
    Sigmoid();
    ~Sigmoid();

    Tensor* forward(Tensor *input);
    Tensor* backward(Tensor *grad);

private:
    Tensor *output = nullptr;
};

#endif