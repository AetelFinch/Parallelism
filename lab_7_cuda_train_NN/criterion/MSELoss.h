#ifndef MSELOSS_H
#define MSELOSS_H

#include "../Tensor.h"

class MSELoss
{
public:
    dtype get_loss(Tensor *predict, Tensor *label);
    Tensor* get_grad();

private:
    Tensor *predict;
    Tensor *label;
};

#endif