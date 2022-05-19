#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include <vector>
#include "../Tensor.h"

class BaseLayer
{
public:
    virtual Tensor* forward(Tensor *input) = 0;
    virtual Tensor* backward(Tensor *grad) = 0;

    virtual std::vector<Tensor*> get_state()
    {
        std::vector<Tensor*> vd(0);
        return vd;
    }

    virtual std::vector<Tensor*> get_grads()
    {
        std::vector<Tensor*> vd(0);
        return vd;
    }
};

#endif
