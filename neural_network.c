#include <stdio.h>
#define MAT_IMPLEMENTATION
#include "neural_network.h"

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float sigmoidf(float x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

int nn_init(NN *neural_network, size_t count, size_t *layers, float rate) {
  if (count == 0)
    return -1;
  neural_network->layer_count = count;
  neural_network->learning_rate = rate;
  neural_network->layers = malloc(sizeof(layer) * count);
  if (!layers) 
    return -1;

  for(size_t i = 0; i < count; i++) {
    neural_network->layers[i].input_size = (i == 0) ? layers[0] : layers[i-1];
    neural_network->layers[i].output_size= layers[i];

    layer *current_layer = &neural_network->layers[i];
    mat_init(&current_layer->ws, current_layer->input_size, current_layer->output_size);
    mat_init(&current_layer->bs, 1, current_layer->output_size);
    mat_init(&current_layer->as, 1, current_layer->output_size);
    mat_randomize(&current_layer->ws, 0, 0.05);
    mat_randomize(&current_layer->bs, 0, 0.05);
  }
  return 0;
}

int nn_deinit(NN *neural_network) {
  for (size_t i = 0; i < neural_network->layer_count; i++) {
    mat_deinit(&neural_network->layers[i].ws);
    mat_deinit(&neural_network->layers[i].bs);
    mat_deinit(&neural_network->layers[i].as);
    neural_network->layers[i].input_size = 0;
    neural_network->layers[i].output_size = 0;
  }
  free(neural_network->layers);
  neural_network->layers = 0;
  neural_network->layer_count = 0;
  neural_network->learning_rate = 0;
  return 0;
}

NN *nn_alloc(size_t count, size_t *layers, float rate) {
  NN *neural_network = malloc(sizeof(NN)); 
  if (!neural_network)
    return NULL;

  nn_init(neural_network, count, layers, rate);
  return neural_network;
}

int NN_free(NN *neural_network) {
  nn_deinit(neural_network);
  free(neural_network);
  neural_network = 0;
  return 0;
}

int forward_propagation(NN *network) {
  for (size_t i = 1; i < network->layer_count; i++) {
    mat_multiply(&network->layers[i].as, 
        &network->layers[i -1].as, &network->layers[i].ws);
    mat_sum(&network->layers[i].as, &network->layers[i].bs);
    mat_activate(&network->layers[i].as, sigmoidf);
  }
  return 0;
}

float cost(NN *network, Mat ti, Mat to) {
  Mat x;
  Mat y_expected; 
  Mat *y_computed;
  if (mat_init(&x, 1, ti.cols))
    perror("mat init x");
  if (mat_init(&y_expected, 1, to.cols))
    perror("mat init y_expected");
  float cost = 0.0f;

  for (size_t i = 0; i < ti.rows ; i++) {
    mat_row(&x, &ti, i);
    mat_inplace_copy(&network->layers[0].as, &x);
    forward_propagation(network);
    y_computed = &network->layers[network->layer_count - 1].as;
    mat_row(&y_expected, &to, i);
    for (size_t j = 0; j < to.cols; j++) {
      float dist =  MAT_AT(y_expected, 0, j) - MAT_AT(*y_computed, 0, j);
      cost += dist * dist;
    }
  }

  mat_deinit(&x);
  mat_deinit(&y_expected);
  cost /= ti.rows;
  return cost;
}


int gradient_init(gradient *grad, NN *neural_network) {
  grad->layer_count = neural_network->layer_count;
  grad->layers = malloc(sizeof(layer) * grad->layer_count);

  if (!grad->layers)
    return -1;

  for (size_t i = 0; i < grad->layer_count; i++) {
    size_t output_size = neural_network->layers[i].output_size;
    size_t input_size = neural_network->layers[i].input_size;
    grad->layers[i].input_size = input_size;
    grad->layers[i].output_size = output_size;
    mat_init(&grad->layers[i].ws, input_size, output_size); 
    mat_init(&grad->layers[i].bs, 1, output_size); 
    grad->layers[i].as.es = 0;
  }

  return 0;
}

int gradient_deinit(gradient *gradient) {
  for (size_t i = 0; i < gradient->layer_count; i++) {
    mat_deinit(&gradient->layers[i].ws);
    mat_deinit(&gradient->layers[i].bs);
  }

  free(gradient->layers);
  gradient->layers = 0;
  gradient->layer_count = 0;
  return 0;
}

gradient *gradient_alloc(NN *neural_network) {
  gradient *grad = malloc(sizeof(gradient));
  if (!grad)
    return NULL;
  if (!gradient_init(grad, neural_network))
    return NULL;
  return grad;
}

int gradient_free(gradient *gradient) {
  if (gradient_deinit(gradient))
    return -1;

  free(gradient);
  return 0;
}

void gradient_compute(NN *neural_network, gradient *grad, 
    float eps, Mat ti, Mat to) {
  
  // have to go over only on the internal layers
  // so we skip layer[0]  
  float c = cost(neural_network, ti, to);
  float saved = 0.0f;
  for (size_t i = 1; i < neural_network->layer_count; i++) {
    for (size_t row = 0; row < neural_network->layers[i].ws.rows; row++) {
      for (size_t col = 0; col < neural_network->layers[i].ws.rows; col++) {
        saved = MAT_AT(neural_network->layers[i].ws, row, col);
        MAT_AT(neural_network->layers[i].ws, row, col) += eps;
        MAT_AT(grad->layers[i].ws, row, col) = 
          (cost(neural_network, ti, to) -c) / eps; 
        MAT_AT(neural_network->layers[i].ws, row, col) = saved;
      }
    }

    for (size_t col = 0; col < neural_network->layers[i].bs.cols; col++) {
      saved = MAT_AT(neural_network->layers[i].bs, 0, col);
      MAT_AT(neural_network->layers[i].bs, 0, col) += eps;
      MAT_AT(grad->layers[i].bs, 0, col) = 
        (cost(neural_network, ti, to) - c) / eps; 
      MAT_AT(neural_network->layers[i].bs, 0, col) = saved;
    }
  }
}

void learn(NN *network, gradient *grad, Mat ti, Mat to) {
  gradient_compute(network, grad, 1e-3, ti, to);
  for (size_t i = 0; i < network->layer_count; i++) {
    for (size_t row = 0; row < network->layers[i].ws.rows; row++) {
      for (size_t col = 0; col < network->layers[i].ws.cols; col++)
        MAT_AT(network->layers[i].ws, row, col) -= 
          network->learning_rate * MAT_AT(grad->layers[i].ws, row, col);
    }

    for (size_t col = 0; col < network->layers[i].bs.cols; col++)
      MAT_AT(network->layers[i].bs, 0, col) -= 
        network->learning_rate * MAT_AT(grad->layers[i].bs, 0, col);
  }
}
