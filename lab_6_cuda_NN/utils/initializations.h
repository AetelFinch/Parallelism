#ifndef RAND_H
#define RAND_H

#include "../Tensor.h"

Tensor* get_normal_distribution(int height, int width, double mean, double stddev);

Tensor* get_kaiming_distribution(int height, int width);

Tensor* get_zeros(int height, int width);

#endif
