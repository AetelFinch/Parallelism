#include <cuda_runtime.h>
#include <iostream>

#include "Sigmoid.h"

__device__ float sigmoid(float x)
{
	return 1.0f / (1 + __expf(-x));
}

__global__ void sigmoidActivationForward(dtype *in, dtype *out, int height, int width)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < height * width)
    {
		out[idx] = sigmoid(in[idx]);
	}
}

Sigmoid::Sigmoid() 
{ 
    this->output = new Tensor();
}

Sigmoid::~Sigmoid()
{
    cudaFree(this->output);
}

Tensor* Sigmoid::forward(Tensor *input)
{
    this->output = new Tensor();
    cudaMalloc(&(this->output->tensor), input->height * input->width * sizeof(dtype));
    this->output->height = input->height;
    this->output->width = input->width;

    dim3 BS(256);
    dim3 GS((input->height * input->width + BS.x - 1) / BS.x);

    sigmoidActivationForward<<<GS, BS>>>(input->tensor, output->tensor, input->height, input->width);

    return output;
}