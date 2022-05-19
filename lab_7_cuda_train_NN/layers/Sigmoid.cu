#include <cuda_runtime.h>
#include <iostream>

#include "Sigmoid.h"
#include "utils/initializations.h"

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

__global__ void sigmoidActivationBackward(dtype *in, dtype *grad, dtype *out, int height, int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < height * width)
    {
		out[idx] = grad[idx] * in[idx] * (1 - in[idx]);
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
    this->output = get_empty_device(input->height, input->width);

    dim3 BS(256);
    dim3 GS((input->height * input->width + BS.x - 1) / BS.x);

    sigmoidActivationForward<<<GS, BS>>>(input->tensor, output->tensor, input->height, input->width);

    return output;
}

Tensor* Sigmoid::backward(Tensor *grad)
{
    Tensor *prev_grad = get_empty_device(grad->height, grad->width);

    dim3 BS(256);
    dim3 GS((output->height * output->width + BS.x - 1) / BS.x);

    sigmoidActivationBackward<<<GS, BS>>>(output->tensor, grad->tensor, prev_grad->tensor, output->height, output->width);

    return prev_grad;
}