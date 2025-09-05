#ifndef MODELFIT_H_
#define MODELFIT_H_

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include "RNG.h"

#define LOSS_MSE 0x0
#define LOSS_MAE 0x1

// float Model(float *prev_inputs, float *prev_outputs, float *parameters)
// float ModelGradient(float *prev_inputs, float *prev_outputs, float *parameters, int param)
// void Termination(float *parameters, float err)

typedef float (*Model)(float*, float*, float*);
typedef float (*ModelGradient)(float*, float*, float*, int32_t parameter);
typedef int (*Termination)(float*, float);

typedef struct {
    float *x;
    float *y;
    int32_t size;
} TrainingExample ;

int gradient_descent(Model model, // model function
    ModelGradient gradients, // model gradients (size of params)
    float *parameters, // inital guess
    TrainingExample *examples, // training examples
    int32_t params, // parameter amount
    int32_t examples_amount, // examples amount
    int32_t batch_size, // batch size (lower or equal to sum of examples size, if zero or invalid will use full batch)
    int32_t input_memory, // memory size of inputs
    int32_t output_memory, // memory size of outputs
    Termination term, // a function determining if to terminate
    int8_t loss, // loss function type
    float rate,
    float alpha,
    int adaptive,
    RNG *rng
) {
    float *rates = (float*) malloc(sizeof(float) * params);

    if (rates == NULL) {
        return -1;
    }

    float *gradients_arr = (float*) malloc(sizeof(float) * params);

    if (gradients_arr == NULL) {
        free(rates);
        return -1;
    }

    float *x = (float*) malloc(sizeof(float) * input_memory);

    if (x == NULL) {
        free(rates);
        free(gradients_arr);
        return -1;
    }

    float *y = (float*) malloc(sizeof(float) * output_memory);

    if (y == NULL && output_memory != 0) {
        free(rates);
        free(gradients_arr);
        free(x);
        return -1;
    }


    for (int32_t i = 0; i < params; i++) {
        rates[i] = 0;
        gradients_arr[i] = 0;
    }

    for (int i = 0; i < input_memory; i++)
        x[i] = 0;

    for (int i = 0; i < output_memory; i++)
        y[i] = 0;

    int32_t *examples_index;
    int32_t *datapoint_index;

    int64_t total_size = 0; // count total point amount

    for (int32_t e = 0; e < examples_amount; e++) {
        total_size += examples[e].size;
    }

    int32_t memory_size = 0;

    if (input_memory > output_memory)
        memory_size = input_memory;
    else
        memory_size = output_memory;


    // initialize batch indices

    int32_t active_batch_size = batch_size;

    if (batch_size >= total_size || 0 >= batch_size)
        active_batch_size = total_size - (memory_size + 1);

    examples_index = (int32_t*) malloc(sizeof(int32_t) * active_batch_size);

    if (examples_index == NULL) {
        free(rates);
        free(gradients_arr);
        free(x);
        free(y);
        return -1;
    }

    datapoint_index = (int32_t*) malloc(sizeof(int32_t) * active_batch_size);

    if (datapoint_index == NULL) {
        free(rates);
        free(gradients_arr);
        free(x);
        free(y);
        free(examples_index);
        return -1;
    }

    while (1) {
        for (int i = 0; i < params; i++) {
            gradients_arr[i] = 0;
        }

        // generate batches index

        if (batch_size >= total_size || 0 >= batch_size) {
            // SGD / minibatch
            for (int i = 0; i < active_batch_size; i++) {
                examples_index[i] = rng_pcg32(rng) % examples_amount;
                datapoint_index[i] = rng_pcg32(rng) % (examples[examples_index[i]].size - memory_size - 1) + (memory_size + 1);
            }
        } else {
            // batch gradient descent
            int i = 0;
            for (int n = 0; n < examples_amount; n++) {
                for (int k = memory_size + 1; k < examples[n].size; k++) {
                    if (i >= active_batch_size) break;
                    examples_index[i] = n;
                    datapoint_index[i] = k;
                    i++;
                }
            }
        }

        float err = 0;

        for (int e = 0; e < active_batch_size; e++) {
            for (int n = 0; n < input_memory; n++) {
                int32_t idx = datapoint_index[e] - n;
                if (idx < 0 || idx >= examples[examples_index[e]].size) {
                    // handle error or skip
                } else {
                    x[n] = examples[examples_index[e]].x[datapoint_index[e] - n];
                }
            }

            for (int n = 0; n < output_memory; n++) {
                int32_t idx = datapoint_index[e] - n;
                if (idx < 0 || idx >= examples[examples_index[e]].size) {
                    // handle error or skip
                } else {
                    y[n] = examples[examples_index[e]].y[datapoint_index[e] - n - 1];
                }
            }

            float eval = model(x, y, parameters);

            for (int p = 0; p < params; p++) {
                float grad = gradients(x, y, parameters, p);
                float loss_grad = 0;
                    
                // MSE: 2 * (f(x) - y) * f'(x)
                // MAE: (f(x) - y) * f'(x) / abs(f(x) - y)

                float true_y = examples[examples_index[e]].y[datapoint_index[e] ];

                if (loss == LOSS_MSE) {
                    loss_grad = 2 * (eval - true_y) * grad;
                    err = (eval - true_y) * (eval - true_y);
                } else if (loss == LOSS_MAE) {
                    if ((eval - true_y) * (eval - true_y) > 1e-6) {
                        loss_grad = (eval - true_y) * grad / ( ( (eval - true_y) > 0) ? (eval - true_y) : -(eval - true_y) );
                        err = ( ( (eval - true_y) > 0) ? (eval - true_y) : -(eval - true_y) );
                    } else {
                        loss_grad = 0;
                        err = 0;
                    }
                }
                gradients_arr[p] += loss_grad / (float) active_batch_size;
            }
        }

        for (int p = 0; p < params; p++) {
            rates[p] = alpha * rates[p] + (1 - alpha) * gradients_arr[p] * gradients_arr[p];
            float actual_rate = rate;
            if (adaptive == 1) {
                actual_rate /= sqrt(rates[p] + 1e-6);
            }

            parameters[p] -= gradients_arr[p] * actual_rate;
        }

        if (term(parameters, err / (float) active_batch_size) ) {
            free(examples_index);
            free(datapoint_index);

            free(rates);
            free(gradients_arr);

            free(x);
            free(y);
            break;
        }
    }
    return 0;
}

#endif