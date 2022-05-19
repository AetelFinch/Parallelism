#include <iostream>
#include "model.h"
#include "layers/Linear.h"
#include "layers/Sigmoid.h"
#include "utils/saver.h"


Model::Model()
{
    layers = new std::vector<BaseLayer*>;

    layers->push_back(new Linear(32 * 32, 16 * 16, true));
    layers->push_back(new Sigmoid());
    layers->push_back(new Linear(16 * 16, 4 * 4, true));
    layers->push_back(new Sigmoid());
    layers->push_back(new Linear(4 * 4, 1, true));
    layers->push_back(new Sigmoid());
}

Model::~Model()
{
    delete layers;
}

Tensor* Model::forward(Tensor* x)
{
    for(auto layer : *layers)
    {
        x = layer->forward(x);
    }
    return x;
}

void Model::backward(Tensor *grad)
{
    std::vector<BaseLayer*>::reverse_iterator rit;

    for(rit = (*layers).rbegin(); rit != (*layers).rend(); ++rit)
    {
        grad = (*rit)->backward(grad);
    }
}

void Model::save(std::string filename)
{
    int num_layer = 0;
    for (auto layer : *layers)
    {
        std::vector<Tensor*> state = layer->get_state();
        if (state.capacity() == 0) continue;

        save_state(state.at(0)->copy2cpu_(), filename + std::string("_weights_") + std::to_string(num_layer) + std::string(".pth"));
        save_state(state.at(1)->copy2cpu_(), filename + std::string("_bias_") + std::to_string(num_layer) + std::string(".pth"));

        ++num_layer;
    }
}

std::vector<std::vector<Tensor*>> Model::get_params()
{
    std::vector<std::vector<Tensor*>> params;
    for (auto layer : *layers)
    {
        std::vector<Tensor*> layer_params = layer->get_state();
        if (layer_params.capacity() == 0) continue;

        params.push_back(layer_params);
    }
    return params;
}

std::vector<std::vector<Tensor*>> Model::get_grads()
{
    std::vector<std::vector<Tensor*>> grads;
    for (auto layer : *layers)
    {
        std::vector<Tensor*> layer_grads = layer->get_grads();
        if (layer_grads.capacity() == 0) continue;

        grads.push_back(layer_grads);
    }
    return grads;
}