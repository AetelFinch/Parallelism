#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <string>

#include "layers/BaseLayer.h"

class Model
{
public:
    Model();
    ~Model();

    Tensor* forward(Tensor *input);
    void backward(Tensor *grad);
    
    void save(std::string filename);
    std::vector<std::vector<Tensor*>> get_params();
    std::vector<std::vector<Tensor*>> get_grads();
private:
    std::vector<BaseLayer*> *layers;
};

#endif