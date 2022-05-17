#include <random>
#include <cmath>
#include <iostream>

#include "initializations.h"

Tensor* get_normal_distribution(int height, int width, double mean, double stddev)
{
    std::mt19937 generator;
    std::normal_distribution<double> normal(mean, stddev);

    Tensor *output = new Tensor();
    output->height = height;
    output->width = width;
    output->tensor = (dtype*)malloc(sizeof(dtype) * height * width);
    for (int i = 0; i < height * width; ++i)
    {
        output->tensor[i] = normal(generator);
    }

    return output;
}

Tensor* get_kaiming_distribution(int height, int width)
{
    Tensor *output = get_normal_distribution(height, width, 0.0, 1.0);
    double alpha = sqrt(2.0 / (double)height);

    for (int i = 0; i < height * width; ++i)
    {
        output->tensor[i] *= alpha;
    }

    return output;
}

Tensor* get_zeros(int height, int width)
{
    Tensor *output = new Tensor();
    output->height = height;
    output->width = width;
    output->tensor = (dtype*)calloc(sizeof(dtype), height * width);

    return output;
}