#include <iostream>

#include "Linear.h"
#include "../utils/initializations.h"
#include "../utils/saver.h"

Linear::Linear(int in_features, int out_features, bool bias)
{
    cublasCreate(&(this->handle));

    this->in_features = in_features;
    this->out_features = out_features;

    Tensor *w_h = get_kaiming_distribution(in_features, out_features);
    this->weights = w_h->copy2gpu();

    delete w_h;

    Tensor *b_h;
    if (bias)
    {
        b_h = get_normal_distribution(1, out_features, 0.0, 1.0);  
    }
    else
    {
        b_h = get_zeros(1, out_features);
    }
    this->bias = b_h->copy2gpu();
    delete b_h;
}

Linear::~Linear()
{
    cublasDestroy(this->handle);

    cudaFree(this->bias->tensor);
    cudaFree(this->weights->tensor);
    cudaFree(this->output->tensor);
}

Tensor* Linear::forward(Tensor* input)
{
    this->output = this->bias->gpu2gpu();

    float k = 1.0;

    cublasSgemm( // C = alpha * AB + beta * C
        handle,
        CUBLAS_OP_N, // transA
        CUBLAS_OP_N, // transB
        input->height, // num rows in A
        this->out_features, // num columns in B and C
        this->in_features, // num rows in B
        &k, // alpha
        input->tensor, // A
        input->height, // The leading dimension of A
        this->weights->tensor, // B
        input->width, // The leading dimension of B
        &k, // beta
        this->output->tensor, // C
        this->bias->height // The leading dimension of C
    );

    return this->output;
}

std::vector<Tensor*> Linear::get_state()
{
    std::vector<Tensor*> state;
    state.push_back(this->weights->copy2cpu());
    state.push_back(this->bias->copy2cpu());
    return state;
}