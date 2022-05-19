#ifndef TENSOR_H
#define TENSOR_H

#include <cstdlib>
#include <cuda_runtime.h>

using dtype = float;

class Tensor
{
public:
    dtype *tensor = nullptr;
    int height;
    int width;
    char device = 'h';

    ~Tensor();

    Tensor* copy2gpu();
    Tensor* copy2cpu();
    Tensor* gpu2gpu();

    Tensor* copy2gpu_();
    Tensor* copy2cpu_();
};

#endif
