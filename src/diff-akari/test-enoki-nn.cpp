// #include <iostream>
// #include <enoki/array.h>
// #include <enoki/cuda.h>
// #include <enoki/autodiff.h>
// #include <enoki/random.h>
// #include <enoki/matrix.h>
// using namespace enoki;

// using FloatC = CUDAArray<float>;
// using UIntC  = CUDAArray<uint32_t>;
// using UIntD  = DiffArray<UIntC>;
// using FloatD = DiffArray<FloatC>;

// using Tensor = FloatD;
// template <class Float>
// Float relu(Float x) {
//     return enoki::max(0.0, x);
// }

// struct FC {
//     float learning_rate = 0.01;
//     FC(size_t in, size_t out) {
//         using RNG = PCG32<FloatD>;
//         {
//             RNG rng(PCG32_DEFAULT_STATE, arange<FloatD>(in));
//             for (size_t i = 0; i < out; i++) {
//                 auto w = rng.next_float32();
//                 weights.push_back(w);
//             }
//         }
//         {
//             RNG rng(PCG32_DEFAULT_STATE, arange<FloatD>(out));
//             bias = rng.next_float32();
//         }
//     }
//     Tensor operator()(const Tensor &x, bool need_grad = true) { return forward(x, true); }
//     Tensor forward(const Tensor &x, bool need_grad) {
//         if (need_grad) {
//             for (size_t i = 0; i < weights.size(); i++) {
//                 requires_gradient(weights[i]);
//             }
//             requires_gradient(bias);
//         }

//         Tensor out = enoki::zero<Tensor>(weights.size());
//         for (size_t i = 0; i < weights.size(); i++) {
//             // out[i] = dot(x, weights[i]);

//             enoki::scatter(out, dot(x, weights[i]) + bias, UIntD(i));
//         }
//         return out;
//     }
//     void backward() {
//         for (size_t i = 0; i < weights.size(); i++) {
//             weights[i] -= gradient(weights[i]) * learning_rate;
//         }
//         bias -= gradient(bias) * learning_rate;
//     }
//     Tensor bias;
//     std::vector<Tensor> weights;
// };

// struct NN {
//     FC fc1, fc2, fc3;
//     NN() : fc1(2, 8), fc2(8, 8), fc3(8, 1) {}
//     Tensor forward(Tensor x) {
//         x = relu(fc1(x));
//         x = relu(fc2(x));
//         x = relu(fc3(x));
//         return x;
//     }
//     void backward() {
//         fc3.backward();
//         fc2.backward();
//         fc1.backward();
//     }
// };

// template <class Float>
// Float target(const Float &x) {
//     // auto x0 = x[UintD(0)];
//     // auto x1 = x[UintD(1)];
//     // return sqrt(x0 * x0,)
//     return sqrt(dot(x, x)) - 0.5;
// }
// int main() {
//     NN nn;
//     using RNG = PCG32<FloatD>;
//     RNG rng(PCG32_DEFAULT_STATE, arange<FloatD>(2));
//     for (int epoch = 0; epoch < 1000; epoch++) {
//         for (int batch = 0; batch < 20; batch++) {
//             auto x    = rng.next_float32();
//             auto y    = nn.forward(x);
//             auto loss = pow(hsum(x - y), 2);
//             backward(loss);
//             std::cout << loss << std::endl;
//         }
//     }
// }
#include <torch/torch.h>
#include <iostream>
#include <random>
struct SimpleNetImpl : torch::nn::Module {
    SimpleNetImpl()
        : fc1(register_module("fc1", torch::nn::Linear(2, 8))), fc2(register_module("fc2", torch::nn::Linear(8, 8))),
          fc3(register_module("fc3", torch::nn::Linear(8, 1))) {}
    torch::Tensor forward(torch::Tensor x) {
        using namespace torch;
        x = relu(fc1(x));
        x = relu(fc2(x));
        x = relu(fc3(x));
        return x;
    }
    torch::nn::Linear fc1, fc2, fc3;
};
TORCH_MODULE(SimpleNet);
int main() {
    std::random_device rd;
    std::uniform_real_distribution<float> dist(-2, 2);
    SimpleNet net;
    auto F = [](float x, float y) { return std::sqrt(x * x + y * y) - 1.0; };
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(0.001));
    for (int epoch = 1; epoch <= 10000; epoch++) {
        net->zero_grad();
        std::vector<std::vector<float>> batch_in;
        std::vector<float> batch_out;
        for (int i = 0; i < 100; i++) {
            auto in = std::vector{dist(rd), dist(rd)};
            batch_out.push_back(F(in[0], in[1]));
            batch_in.push_back(std::move(in));
        }
        auto x = torch::tensor(batch_in);

        auto y             = net->forward(x);
        auto target        = torch::tensor(batch_out);
        torch::Tensor loss = torch::mse_loss(y, target);

        loss.backward();
        opt.step();
        printf("epoch:%d loss:%f\n", epoch, loss.item<float>());
    }
}