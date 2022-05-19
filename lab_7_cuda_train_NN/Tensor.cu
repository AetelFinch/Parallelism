#include "Tensor.h"

Tensor::~Tensor()
{
    if (device == 'h')
        free(tensor);
    else
        cudaFree(tensor);
}

Tensor* Tensor::copy2gpu()
{
    Tensor *output = new Tensor();
    cudaMalloc(&(output->tensor), this->height * this->width * sizeof(dtype));
    cudaMemcpy(output->tensor, this->tensor, this->height * this->width * sizeof(dtype), cudaMemcpyHostToDevice);
    output->height = this->height;
    output->width = this->width;
    output->device = 'd';
    return output;
}

Tensor* Tensor::copy2cpu()
{
    Tensor *output = new Tensor();
    output->tensor = (dtype*)malloc(this->height * this->width * sizeof(dtype));
    cudaMemcpy(output->tensor, this->tensor, this->height * this->width * sizeof(dtype), cudaMemcpyDeviceToHost);
    output->height = this->height;
    output->width = this->width;
    output->device = 'c';
    return output;
}

Tensor* Tensor::gpu2gpu()
{
    Tensor *output = new Tensor();
    cudaMalloc(&(output->tensor), this->height * this->width * sizeof(dtype));
    cudaMemcpy(output->tensor, this->tensor, this->height * this->width * sizeof(dtype), cudaMemcpyDeviceToDevice);
    output->height = this->height;
    output->width = this->width;
    output->device = 'd';
    return output;
}

Tensor* Tensor::copy2gpu_()
{
    Tensor *output = new Tensor();
    cudaMalloc(&(output->tensor), this->height * this->width * sizeof(dtype));
    cudaMemcpy(output->tensor, this->tensor, this->height * this->width * sizeof(dtype), cudaMemcpyHostToDevice);
    output->height = this->height;
    output->width = this->width;
    output->device = 'd';

    delete this;
    return output;
}

Tensor* Tensor::copy2cpu_()
{
    Tensor *output = new Tensor();
    output->tensor = (dtype*)malloc(this->height * this->width * sizeof(dtype));
    cudaMemcpy(output->tensor, this->tensor, this->height * this->width * sizeof(dtype), cudaMemcpyDeviceToHost);
    output->height = this->height;
    output->width = this->width;
    output->device = 'c';

    delete this;
    return output;
}