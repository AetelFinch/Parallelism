#include "Tensor.h"

Tensor* Tensor::copy2gpu()
{
    Tensor *output = new Tensor();
    cudaMalloc(&(output->tensor), this->height * this->width * sizeof(dtype));
    cudaMemcpy(output->tensor, this->tensor, this->height * this->width * sizeof(dtype), cudaMemcpyHostToDevice);
    output->height = this->height;
    output->width = this->width;
    return output;
}

Tensor* Tensor::copy2cpu()
{
    Tensor *output = new Tensor();
    output->tensor = (dtype*)malloc(this->height * this->width * sizeof(dtype));
    cudaMemcpy(output->tensor, this->tensor, this->height * this->width * sizeof(dtype), cudaMemcpyDeviceToHost);
    output->height = this->height;
    output->width = this->width;
    return output;
}

Tensor* Tensor::gpu2gpu()
{
    Tensor *output = new Tensor();
    cudaMalloc(&(output->tensor), this->height * this->width * sizeof(dtype));
    cudaMemcpy(output->tensor, this->tensor, this->height * this->width * sizeof(dtype), cudaMemcpyDeviceToDevice);
    output->height = this->height;
    output->width = this->width;
    return output;
}