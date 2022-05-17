#include <iostream>

#include "utils/initializations.h"
#include "utils/saver.h"
#include "model.h"

int main()
{
    Tensor *input = get_normal_distribution(1, 32 * 32, 0.0, 1.0);

    Tensor* input_d = input->copy2gpu();
    delete input;

    Model model;
    Tensor *out_d = model.forward(input_d);

    Tensor *out = out_d->copy2cpu();
    std::cout << "output = " << out->tensor[0] << "\n";

    save_state(input, "weights/input.pth");
    model.save("weights/exp");

    return 0;
}