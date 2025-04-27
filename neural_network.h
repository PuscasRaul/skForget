#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Mat.h"

#ifndef NN_ALLOC
#define NN_ALLOC malloc
#endif // NN_ALLOC

typedef struct {
  size_t input_size; 
  size_t output_size;
  Mat ws;
  Mat bs;
  Mat as;
} layer;

typedef struct {
  size_t layer_count; // how many layers the neural network contains
  layer *layers; 
  float learning_rate;
} NN;

typedef struct {
  size_t layer_count;
  layer *layers;
} gradient;

// for neural network
int nn_init(NN *neural_network, size_t count, size_t *layers, float rate);
int nn_deinit(NN *neural_network);
NN *nn_alloc(size_t count, size_t *layers, float rate);
int NN_free(NN *neural_network);
int forward_propagation(NN *network);
float cost(NN *network, Mat ti, Mat to);
void learn(NN *network, gradient *grad, Mat ti, Mat to);

// for gradient
int gradient_init(gradient *gradient, NN *neural_network);
int gradient_deinit(gradient *gradient);
gradient *gradient_alloc(NN *neural_network);
int gradient_free(gradient *gradient);
#endif

