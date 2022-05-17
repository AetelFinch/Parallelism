#ifndef SIGMOID_H
#define SIGMOID_H

#include "BaseLayer.h"

class Sigmoid: public BaseLayer
{
public:
    Sigmoid();
    ~Sigmoid();

    Tensor* forward(Tensor *input);

private:
    Tensor *output = nullptr;
};

#endif