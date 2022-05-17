#include <fstream>
#include <iostream>

#include "saver.h"

void save_state(Tensor *input, std::string filename)
{
    std::ofstream file(filename);

    file << input->height << " " << input->width << "\n";

    for (int i = 0; i < input->height * input->width; ++i)
    {
        file << input->tensor[i] << " ";
    }
}