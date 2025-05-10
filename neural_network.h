#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Mat.h"

#ifndef NN_ALLOC
#define NN_ALLOC malloc
#endif // NN_ALLOC

#ifndef NN_PRINT
#define NN_PRINT(nn) nn_print(nn, #nn) 
#endif

#ifndef NN_INPUT
#define NN_INPUT(nn) nn.layers[0].as
#endif

#ifndef NN_OUTPUT
#define NN_OUTPUT(nn) nn->layers[nn->layer_count - 1].as
#endif

#ifndef NN_ACT
#define NN_ACT ACT_SIG
#endif

#ifndef NN_RELU_PARAM
#define NN_RELU_PARAM 0.1f
#endif

typedef enum {
  ACT_SIG,
  ACT_RELU,
  ACT_TANH, 
  ACT_SIN,
} Act;

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

/**
 * @brief Initialize a neural network with specified architecture
 * @param neural_network Pointer to the neural network to initialize
 * @param count Number of layers
 * @param layers Array of layer sizes (including input and output layers)
 * @param rate Learning rate
 * @return 0 on success, -1 on failure
 */
int nn_init(NN *neural_network, size_t count, size_t *layers, float rate);

/**
 * @brief Free resources associated with a neural network
 * @param neural_network Pointer to the neural network to deinitialize
 * @return 0 on success, -1 on failure
 */
int nn_deinit(NN *neural_network);

/**
 * @brief Allocate and initialize a new neural network
 * @param count Number of layers
 * @param layers Array of layer sizes (including input and output layers)
 * @param rate Learning rate
 * @return Pointer to allocated neural network or NULL on failure
 */
NN *nn_alloc(size_t count, size_t *layers, float rate);

/**
 * @brief Free a neural network allocated with nn_alloc
 * @param neural_network Pointer to the neural network to free
 * @return 0 on success, -1 on failure
 */
int NN_free(NN *neural_network);

/**
 * @brief Perform forward propagation through the network
 * @param network Pointer to the neural network
 * @return 0 on success, -1 on failure
 * @note Assumes the input layer (layer 0) activations are already set
 */
int forward_propagation(NN *network);

/**
 * @brief Perform backward propagation through the network
 * @param network Pointer to neural network
 * @param grad Pointer to gradient structure
 * @param ti training data for input
 * @param to trainig data for output
 * @return 0 on success, -1 on failure
 */
int backward_propagation(NN *network, gradient *grad, Mat ti, Mat to);

/**
 * @brief Calculate the mean squared error cost for the network
 * @param network Pointer to the neural network
 * @param ti Training input matrix (rows = samples, cols = input features)
 * @param to Training output matrix (rows = samples, cols = output features)
 * @return The calculated cost (mean squared error)
 */
float MSE(NN *network, Mat ti, Mat to);

/**
 * @brief Update network weights and biases using the computed gradients
 * @param network Pointer to the neural network to update
 * @param grad Pointer to gradients
 * @param ti Training input matrix
 * @param to Training output matrix
 * @return 0 on success, -1 on failure
 */
int finite_diff(NN *network, gradient *grad, Mat ti, Mat to);

int learn(NN *network, Mat ti, Mat to, size_t epochs, int (*fn)(NN *network,gradient *grad, Mat ti, Mat to));

void nn_print(const NN * const network, char *name);

// FOR GRADIENT

/**
 * @brief Initialize a gradient structure for a neural network
 * @param gradient Pointer to the gradient to initialize
 * @param neural_network Pointer to the associated neural network
 * @return 0 on success, -1 on failure
 */
int gradient_init(gradient *gradient, NN *neural_network);

/**
 * @brief Free resources associated with a gradient
 * @param gradient Pointer to the gradient to deinitialize
 * @return 0 on success, -1 on failure
 */
int gradient_deinit(gradient *gradient);

/**
 * @brief Allocate and initialize a new gradient for a neural network
 * @param neural_network Pointer to the associated neural network
 * @return Pointer to allocated gradient or NULL on failure
 */
gradient *gradient_alloc(NN *neural_network);

/**
 * @brief Free a gradient allocated with gradient_alloc
 * @param gradient Pointer to the gradient to free
 * @return 0 on success, -1 on failure
 */
int gradient_free(gradient *gradient);
#endif

