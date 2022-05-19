#include "MSELoss.h"
#include "../utils/initializations.h"

#include <iostream>

__global__ void mseLossBackward(dtype *predict, dtype *label, dtype *grad)
{
    grad[0] = 2 * (predict[0] - label[0]);
}

dtype MSELoss::get_loss(Tensor *predict, Tensor *label)
{
    this->predict = predict;
    this->label = label;

    Tensor *predict_h = predict->copy2cpu();
    Tensor *label_h = label->copy2cpu();

    float diff = label_h->tensor[0] - predict_h->tensor[0];
    return diff * diff;
}

Tensor* MSELoss::get_grad()
{
    Tensor *grad = get_empty_device(1, 1);

    mseLossBackward<<<1, 1>>>(predict->tensor, label->tensor, grad->tensor);

    delete label;

    return grad;
}