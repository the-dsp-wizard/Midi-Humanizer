#include <iostream>
#include <time.h>
#include "ModelFit.h"
#include <filesystem>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>

namespace fs = std::filesystem;


// neuron_n' (w.r.t. bias_n) = output_weight_n * f'(Wn * Xn + bn)
// neuron_n' (w.r.t. weight_nm) = output_weight_n * x_nm * f'(Wn * Xn + bn)
// output_neuron' (w.r.t. output_bias) = 1
// output_neuron' (w.r.t. output_weight_n) = f(Wn * Xn + bn)

const int input_memory = 6;
const int output_memory = 6;
const int out_neurons = 4;

// parameters: out_neurons * (inputs + 1) + out_neurons + 1

float activation(float x) {
    return tanh(x);
}

float activation_deriv(float x) {
    return 1 - tanh(x) * tanh(x);
}

int epoch = 0;
int max_epoch = 20000;

int term(float *parameters, float err) {
    // if (epoch % 100 == 0) printf("Err: %f Progress: %f%%, Epoch: %d\n", err, 100 * (float) epoch / (float) max_epoch, epoch);
    epoch++;
    return (epoch > max_epoch);
}

float neuralnet_model(float *prev_inputs, float *prev_outputs, float *parameters) {
    float outs[out_neurons] = {};

    for (int output = 0; output < out_neurons; output++) {
        for (int param = 0; param < (input_memory + output_memory + 1); param++) {
            if (param < input_memory)
                outs[output] += prev_inputs[param] * parameters[(input_memory + output_memory + 1) * output + param];
            else if (param < (input_memory + output_memory) )
                outs[output] += prev_outputs[param - input_memory] * parameters[(input_memory + output_memory + 1) * output + param];
            else
                outs[output] += parameters[(input_memory + output_memory + 1) * output + param];
        }

        outs[output] = activation(outs[output]);
    }

    for (int output = 0; output < out_neurons; output++) {
        outs[output] *= parameters[output + out_neurons * (input_memory + output_memory + 1)];
    }

    float sum = 0;

    for (int output = 0; output < out_neurons; output++) {
        sum += outs[output];
    }

    return sum + parameters[out_neurons + out_neurons * (input_memory + output_memory + 1)];
}

float grad_table[out_neurons];
float eval_table[out_neurons];

float neuralnet_model_grad(float *prev_inputs, float *prev_outputs, float *parameters, int32_t param) {
    if (param < (input_memory + output_memory + 1) * out_neurons) {
        int32_t neuron = param % (input_memory + output_memory + 1);
        int32_t output = (param - neuron) / (input_memory + output_memory + 1);
        float weight = parameters[output + out_neurons * (input_memory + output_memory + 1)];
        float input = 1;

        if (neuron < input_memory)
            input = prev_inputs[neuron];
        else if (neuron < (input_memory + output_memory) )
            input = prev_outputs[neuron - input_memory];

        float out = 0;

        for (int i = 0; i < (input_memory + output_memory + 1); i++) {
            if (i < input_memory)
                out += prev_inputs[i] * parameters[(input_memory + output_memory + 1) * output + i];
            else if (i < (input_memory + output_memory) )
                out += prev_outputs[i - input_memory] * parameters[(input_memory + output_memory + 1) * output + i];
            else
                out += parameters[(input_memory + output_memory + 1) * output + i];
        }

        out = activation_deriv(out);

        return out * input * weight;
    } else if (param == out_neurons + out_neurons * (input_memory + output_memory + 1)) {
        return 1;
    } else {
        int32_t output = param - out_neurons * (input_memory + output_memory + 1);

        float eval = parameters[(input_memory + output_memory + 1) * output + (input_memory + output_memory)];

        for (int i = 0; i < input_memory + output_memory; i++) {
            if (i < input_memory)
                eval += prev_inputs[i] * parameters[(input_memory + output_memory + 1) * output + i];
            else if (i < (input_memory + output_memory) )
                eval += prev_outputs[i - input_memory] * parameters[(input_memory + output_memory + 1) * output + i];
        }

        return activation(eval);
    }
}

float neuralnet_model_grad_memo(float *prev_inputs, float *prev_outputs, float *parameters, int32_t param) {
    if (param == 0) {
        for (int out = 0; out < out_neurons; out++) {
            float sum = 0;
            for (int n = 0; n < (input_memory + output_memory + 1); n++) {
                if (n < input_memory)
                    sum += prev_inputs[n] * parameters[(input_memory + output_memory + 1) * out + n];
                else if (n < (input_memory + output_memory) )
                    sum += prev_outputs[n - input_memory] * parameters[(input_memory + output_memory + 1) * out + n];
                else
                    sum += parameters[(input_memory + output_memory + 1) * out + n];
            }    

            eval_table[out] = activation(sum);
            grad_table[out] = activation_deriv(sum);
        }
    }
    if (param < (input_memory + output_memory + 1) * out_neurons) {
        int32_t neuron = param % (input_memory + output_memory + 1);
        int32_t output = (param - neuron) / (input_memory + output_memory + 1);
        float weight = parameters[output + out_neurons * (input_memory + output_memory + 1)];
        float input = 1;

        if (neuron < input_memory)
            input = prev_inputs[neuron];
        else if (neuron < (input_memory + output_memory) )
            input = prev_outputs[neuron - input_memory];

        return grad_table[output] * input * weight;
    } else if (param == out_neurons + out_neurons * (input_memory + output_memory + 1)) {
        return 1;
    } else {
        int32_t output = param - out_neurons * (input_memory + output_memory + 1);

        return eval_table[output];
    }
}


float sign(float x) {
    if (x == 0) return 0;

    return (x > 0) ? 1 : -1;
}


float generate_gaussian(float mean, float stddev) {
    float uniform = (float) rand() / RAND_MAX;

    float a = -0.1 * stddev;
    float b = 0.1 * stddev;

    float A = -a; // create brackets
    float B = b;
    float c;

    for (int i = 0; i < 50; i++) {
        c = (A + B) * 0.5;
        float eps = (A - mean) / stddev;
        float alpha = (a - mean) / stddev;
        float beta = (b - mean) / stddev;
        float z = 0.5 * (1 + erf(beta / sqrt(2) ) ) - 0.5 * (1 + erf(alpha / sqrt(2) ) );

        float f_a = (0.5 * (1 + erf(eps / sqrt(2) ) ) - 0.5 * (1 + erf(alpha / sqrt(2) ) ) ) / z;
        eps = (c - mean) / stddev;
        float f_c = (0.5 * (1 + erf(eps / sqrt(2) ) ) - 0.5 * (1 + erf(alpha / sqrt(2) ) ) ) / z;

        if (sign(f_a) == sign(f_c)) A = c;
        else B = c;
    }

    return c;
}

int main() {
    TrainingExample examples[13];

    int in_len[13];

    for (int i = 0; i <= 12; i++) {
        char name[100];
        std::sprintf(name, "dataset/ex%d_in.csv", i);

        std::ifstream file(name);
        if (!file.is_open()) {
            std::cerr << "Failed to open file.\n";
            return 1;
        }

        std::string line;

        int len = 0;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;

            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }

            for (const auto& item : row) {
                try {
                    float num = std::stof(item);
                    len++;
                } catch (...) {
                }
            }

        }

        in_len[i] = len;

        examples[i].x = (float*) calloc(len, sizeof(float) );
        if (examples[i].x == NULL) return 1;

        int acc = 0;

        file.close();
        file.open(name);
        if (!file.is_open()) {
            std::cerr << "Failed to reopen file.\n";
            return 1;
        }

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;

            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }
        
            for (const auto& item : row) {
                try {
                    float num = std::stof(item);
                    if (acc < len) examples[i].x[acc] = num;
                    acc++;
                } catch (...) {
                }
            }

        }

        file.close();
    }

    for (int i = 0; i <= 12; i++) {
        char name[100];
        std::sprintf(name, "dataset/ex%d_out.csv", i);

        std::ifstream file(name);
        if (!file.is_open()) {
            std::cerr << "Failed to open file.\n";
            return 1;
        }

        std::string line;
        int len = 0;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;

            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }

            for (const auto& item : row) {
                try {
                    float num = std::stof(item);
                    len++;
                } catch (...) {
                }
            }

        }

        if (len > in_len[i]) len = in_len[i];

        examples[i].y = (float*) calloc(len, sizeof(float) );
        if (examples[i].y == NULL) return 1;

        file.close();
        file.open(name);
        if (!file.is_open()) {
            std::cerr << "Failed to reopen file.\n";
            return 1;
        }

        int acc = 0;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<std::string> row;

            while (std::getline(ss, cell, ',')) {
                row.push_back(cell);
            }

            for (const auto& item : row) {
                try {
                    float num = std::stof(item);
                    if (acc < len) examples[i].y[acc] = num;
                    acc++;
                } catch (...) {
                }
            }
        }

        file.close();

        examples[i].size = in_len[i];
    }

    RNG rng;
    rng_seed(&rng, 3677828);

    int N = out_neurons * (input_memory + output_memory + 1) + out_neurons + 1;

    float parameters[N];
    for (int i = 0; i < N; i++) parameters[i] = 0.5;

    clock_t start = clock();

    int err = gradient_descent(neuralnet_model, // model function
        neuralnet_model_grad_memo, // model gradients (size of params)
        parameters, // inital guess
        examples, // training examples
        N, // parameter amount
        10, // examples amount
        15000, // batch size (lower or equal to sum of examples size, if zero or invalid will use full batch)
        input_memory, // memory size of inputs
        output_memory, // memory size of outputs
        term, // a function determining if to terminate
        LOSS_MSE, // loss function type
        0.005,
        0.99,
        1,
        &rng
    );

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", time_taken);

    float x[input_memory] = {0};
    float y[output_memory] = {0};

    for (int i = 10; i <= 12; i++) {
        float sum = 0;
        float power = 0;
        float mean = 0;
        int size = 0;

        for (int n = 0; n < examples[i].size; n++) {
            power += examples[i].y[n] * examples[i].y[n];
            mean += examples[i].y[n];
        }

        power /= (float) examples[i].size;
        mean /= (float) examples[i].size;

        float variance = 0;

        for (int n = 0; n < examples[i].size; n++) {
            variance += (examples[i].y[n] - mean) * (examples[i].y[n] - mean);
        }

        variance /= (float) examples[i].size;

        printf("import numpy as np\n");
        printf("import matplotlib.pyplot as plt\n");

        printf("y = np.zeros(8000)\n");
        printf("y_hat = np.zeros(8000)\n");

        for (int n = 0; n < examples[i].size; n++) {
            float out = neuralnet_model(x, y, parameters) + generate_gaussian(mean, sqrt(variance) );
            
            for (int k = input_memory - 1; k > 0; k--) x[k] = x[k - 1];
            
            x[0] = examples[i].x[n];

            for (int k = output_memory - 1; k > 0; k--) y[k] = y[k - 1];
            y[0] = out;

            if (i == 10) printf("y[%d] = %f\n", n, examples[i].y[n]);
            if (i == 10) printf("y_hat[%d] = %f\n", n, out);
        }

        printf("plt.plot(y)\n");
        printf("plt.plot(y_hat)\n");
        printf("plt.show()\n");
    }

    for (int i = 0; i <= 12; i++) {
        free(examples[i].x);
        free(examples[i].y);
    }

    return 0;
}
