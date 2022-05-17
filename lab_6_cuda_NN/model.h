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
    void save(std::string filename);
private:
    std::vector<BaseLayer*> *layers;
};

#endif