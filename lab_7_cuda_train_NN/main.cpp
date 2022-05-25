#include <iostream>

#include "utils/initializations.h"
#include "utils/saver.h"
#include "utils/reader.h"
#include "model.h"
#include "criterion/MSELoss.h"
#include "optimizer/SGD.h"

void normalize(Tensor *x)
{
    for (int i = 0; i < x->height * x->width; ++i)
    {
        x->tensor[i] /= 255;
    }
}

int main()
{
    Tensor *digit_1 = read_state("images/digit_1.txt");
    Tensor *digit_8 = read_state("images/digit_8.txt");

    normalize(digit_1);
    normalize(digit_8);

    Tensor *digit_1_d = digit_1->copy2gpu();
    Tensor *digit_8_d = digit_8->copy2gpu();
    std::vector<Tensor*> digits;
    digits.push_back(digit_1_d);
    digits.push_back(digit_8_d);

    delete digit_1;
    delete digit_8;

    Model model;
    MSELoss criterion;
    SGD optimizer(model.get_params(), model.get_grads(), 0.01);

    Tensor *label = get_zeros(1, 1);

    Tensor *label_1_d = label->copy2gpu();
    label->tensor[0] = 1;
    Tensor *label_8_d = label->copy2gpu();
    delete label;

    std::vector<Tensor*> labels;
    labels.push_back(label_1_d);
    labels.push_back(label_8_d);

    std::cout << "initialization completed\n";

    int num_epochs = 1000;
    for (int epoch = 1; epoch <= num_epochs; ++epoch)
    {
        float sum_losses = 0;
        for (int digit_idx = 0; digit_idx < digits.size(); ++digit_idx)
        {
            optimizer.zero_grad();

            Tensor *out_d = model.forward(digits.at(digit_idx));

            float loss = criterion.get_loss(out_d, labels.at(digit_idx));
            sum_losses += loss;

            Tensor *grad = criterion.get_grad();

            model.backward(grad);

            optimizer.step();
        }
        std::cout << "epoch = " << epoch << ": loss = " << sum_losses << "\n";
    }

    return 0;
}