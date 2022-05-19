#include <iostream>

#include "Linear.h"
#include "../utils/initializations.h"
#include "../utils/saver.h"

Linear::Linear(int in_features, int out_features, bool bias)
{
    cublasCreate(&(this->handle));
    isBias = bias;

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

    grad_weights = get_empty_device(weights->height, weights->width);

    if (isBias)
        grad_bias = get_empty_device(this->bias->height, this->bias->width);
}

Linear::~Linear()
{
    cublasDestroy(this->handle);

    cudaFree(this->bias->tensor);
    cudaFree(this->weights->tensor);
    cudaFree(this->input->tensor);
}

Tensor* Linear::forward(Tensor* input)
{
    this->input = input;
    Tensor *output = this->bias->gpu2gpu();

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
        output->tensor, // C
        this->bias->height // The leading dimension of C
    );

    return output;
}

Tensor* Linear::backward(Tensor *grad)
{
    // std::cout << "before info: " << grad->device << " " << grad->height << " " << grad->width << "\n";

    float k = 1.0;

    delete this->grad_weights;
    this->grad_weights = get_empty_device(weights->height, weights->width);
    cublasSgemm( // C = alpha * AB + beta * C
        handle,
        CUBLAS_OP_T, // transA
        CUBLAS_OP_N, // transB
        input->height, // num rows in A
        grad->width, // num columns in B and C
        grad->height, // num rows in B
        &k, // alpha
        input->tensor, // A
        input->width, // The leading dimension of A
        grad->tensor, // B
        grad->height, // The leading dimension of B
        &k, // beta
        this->grad_weights->tensor, // C
        this->grad_weights->height // The leading dimension of C
    );

    delete this->grad_bias;
    this->grad_bias = grad->gpu2gpu();

    Tensor *tmp = get_zeros(grad->height, weights->height);

    Tensor *prev_grad = tmp->copy2gpu_();
    cublasSgemm( // C = alpha * AB + beta * C
        handle,
        CUBLAS_OP_N, // transA
        CUBLAS_OP_T, // transB
        grad->height, // num rows in A
        weights->width, // num columns in B and C
        weights->height, // num rows in B
        &k, // alpha
        grad->tensor, // A
        grad->height, // The leading dimension of A
        weights->tensor, // B
        weights->width, // The leading dimension of B
        &k, // beta
        prev_grad->tensor, // C
        prev_grad->height // The leading dimension of C
    );

    return prev_grad;
}

std::vector<Tensor*> Linear::get_state()
{
    std::vector<Tensor*> state;
    state.push_back(this->weights);

    if (isBias)
        state.push_back(this->bias);
    return state;
}

std::vector<Tensor*> Linear::get_grads()
{
    std::vector<Tensor*> grads;
    grads.push_back(this->grad_weights);

    if (isBias)
        grads.push_back(this->grad_bias);
    return grads;
}