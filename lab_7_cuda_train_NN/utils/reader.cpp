#include <fstream>
#include <iostream>

#include "reader.h"

Tensor* read_state(std::string filename)
{
    std::ifstream opened_file(filename);

    if (!opened_file.is_open())
    {
        std::cout << "file " << filename << " not found\n";
        exit(-1);
    }

    Tensor *x = new Tensor(); 
    opened_file >> x->height >> x->width;
    x->device = 'h';

    x->tensor = (dtype*)malloc(sizeof(dtype) * x->height * x->width);

    for (int i = 0; i < x->height * x->width; ++i)
    {
        opened_file >> x->tensor[i];
    }

    return x;
}