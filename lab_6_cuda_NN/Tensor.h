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

    ~Tensor()
    {
        free(tensor);
    }

    Tensor* copy2gpu();

    Tensor* copy2cpu();

    Tensor* gpu2gpu();
};

#endif
