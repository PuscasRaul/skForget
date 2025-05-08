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
    
  mat_init(&neural_network->layers[0].as, 1, layers[0]);
  mat_init(&neural_network->layers[0].ws, 1, 1);
  mat_init(&neural_network->layers[0].bs, 1, 1);
  neural_network->layers[0].input_size = layers[0];
  neural_network->layers[0].output_size = layers[0];

  for(size_t i = 1; i < count; i++) {
    neural_network->layers[i].input_size = layers[i-1];
    neural_network->layers[i].output_size= layers[i];

    layer *current_layer = &neural_network->layers[i];

    mat_init(&current_layer->ws, current_layer->input_size, current_layer->output_size);
    mat_init(&current_layer->bs, 1, current_layer->output_size);
    mat_init(&current_layer->as, 1, current_layer->output_size);
    mat_randomize(&current_layer->ws, 0, 1);
    mat_fill(&current_layer->bs, 0);
    // mat_randomize(&current_layer->bs, 0, 1);
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
  if (ti.rows != to.rows)
    return INFINITY;
  if (NN_OUTPUT(network).cols != to.cols)
    return INFINITY;

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
      float pred = MAT_AT(*y_computed, 0, j);
      float target = MAT_AT(y_expected, 0, j);
      float distance = pred - target;
      cost += distance * distance;
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
    mat_init(&grad->layers[i].as, 1, output_size);
  }
  return 0;
}

int gradient_deinit(gradient *gradient) {
  for (size_t i = 0; i < gradient->layer_count; i++) {
    mat_deinit(&gradient->layers[i].ws);
    mat_deinit(&gradient->layers[i].bs);
    mat_deinit(&gradient->layers[i].as);
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
  if (gradient_init(grad, neural_network))
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
  float c1, c2;
  for (size_t i = 0; i < neural_network->layer_count; i++) {
    for (size_t row = 0; row < neural_network->layers[i].ws.rows; row++) {
      for (size_t col = 0; col < neural_network->layers[i].ws.cols; col++) {

        float *w = &MAT_AT(neural_network->layers[i].ws, row, col);
        float saved = *w;

        *w = saved + eps;
        c1 = cost(neural_network, ti, to);

        *w = saved - eps;
        c2 = cost(neural_network, ti, to);

        MAT_AT(grad->layers[i].ws, row, col) = (c1 - c2) / (2 * eps);
        *w = saved;
      }
    }

    for (size_t col = 0; col < neural_network->layers[i].bs.cols; col++) {
      float *w = &MAT_AT(neural_network->layers[i].bs, 0, col);
      float saved = *w;

      *w = saved + eps;
      c1 = cost(neural_network, ti, to);

      *w = saved - eps;
      c2 = cost(neural_network, ti, to);

      MAT_AT(grad->layers[i].bs, 0, col) = (c1 - c2) / (2 * eps);
      *w = saved;
    }
  }
}

inline void static gradient_diff(NN *network, gradient * grad) {
  for (size_t i = 1; i < network->layer_count; i++) {
    mat_scalar(&grad->layers[i].ws, (-1) * network->learning_rate);  
    mat_sum(&network->layers[i].ws, &grad->layers[i].ws);           

    mat_scalar(&grad->layers[i].bs, (-1) * network->learning_rate);
    mat_sum(&network->layers[i].bs, &grad->layers[i].bs);
  }
}

int finite_diff(NN *network, gradient *grad, Mat ti, Mat to) {
  gradient_compute(network, grad, 1e-3, ti, to);
  gradient_diff(network, grad);
  return 0;
}

int backward_propagation(NN *network, gradient *grad, Mat ti, Mat to) {
  if (ti.rows != to.rows)
    return -1;

  if (ti.cols != network->layers[0].input_size)
    return -1;

  if (to.cols != network->layers[network->layer_count - 1].as.cols)
    return -1;

  size_t n = ti.rows;
  Mat x;
  Mat *y_computed;
  Mat y_expected;
  mat_init(&x, 1, ti.cols);
  mat_init(&y_expected, 1, to.cols);

  for (size_t layer = 0; layer < grad->layer_count; layer++) {
    mat_fill(&grad->layers[layer].as, 0);
    mat_fill(&grad->layers[layer].ws, 0);
    mat_fill(&grad->layers[layer].bs, 0);
  } 

  // i - current sample
  // l - current layer
  // j - current activation
  // k - previous activation

  for (size_t i = 0 ; i < n; i++) {
    mat_row(&x, &ti, i);
    mat_inplace_copy(&network->layers[0].as, &x);
    forward_propagation(network);

    y_computed = &network->layers[network->layer_count - 1].as;
    mat_row(&y_expected, &to, i);

    // computing the error 

    for (size_t j = 0; j < to.cols; j++) {
      float a = MAT_AT(*y_computed, 0, j);
      float y = MAT_AT(y_expected, 0, j);
      MAT_AT(grad->layers[network->layer_count - 1].as, 0, j) = (a - y);  
    }

    for (int l = network->layer_count - 2; l >= 0; --l) {
      layer *current = &network->layers[l];
      layer *next = &network->layers[l + 1];
      layer *grad_current = &grad->layers[l];
      layer *grad_next = &grad->layers[l + 1];

      for (size_t j = 0; j < current->output_size; j++) {
        float sum = 0.0f;
        for (size_t k = 0; k < next->output_size; k++) {
          float w = MAT_AT(next->ws, j, k);
          float delta_next = MAT_AT(grad_next->as, 0, k);
          sum += w * delta_next;
        }

        float a = MAT_AT(current->as, 0, j);
        MAT_AT(grad_current->as, 0, j) = sum * a * (1 -a);
      }
    }

    for (size_t l = 1; l < network->layer_count; l++) {
      layer *current = &network->layers[l];
      layer *prev = &network->layers[l - 1];
      layer *grad_current = &grad->layers[l];

      for (size_t j = 0; j < current->input_size; j++) {
        for (size_t k = 0; k < current->output_size; k++) {
          float delta = MAT_AT(grad_current->as, 0, k);
          float a_prev = MAT_AT(prev->as, 0, j);
          MAT_AT(grad_current->ws, j, k) += a_prev * delta;
        }
      }

      for (size_t k = 0; k < current->output_size; k++) {
        float delta = MAT_AT(grad_current->as, 0, k);
        MAT_AT(grad_current->bs, 0, k) += delta;
      }
    }
  }

  for (size_t i = 1; i < grad->layer_count; i++) {
    for (size_t j = 0; j < grad->layers[i].ws.rows; j++) {
      for (size_t k = 0; k < grad->layers[i].ws.cols; k++)
        MAT_AT(grad->layers[i].ws, j, k) /= n;
    }

    for (size_t k = 0; k < grad->layers[i].bs.cols; k++)
      MAT_AT(grad->layers[i].bs, 0, k) /= n;
  }
 
  gradient_diff(network, grad);
  mat_deinit(&y_expected);
  mat_deinit(&x);

  return 0;
}

int learn(NN *network, Mat ti, Mat to, size_t epochs, int (*fn)(NN *network, gradient *grad, Mat ti, Mat to)) {
  gradient grad;
  if (gradient_init(&grad, network))
    return -1;

  for (size_t epoch = 0; epoch < epochs; epoch++) {
    fn(network, &grad, ti, to);
  }

  return 0;
}

void nn_print(const NN * const network, char *name) {
  printf("%s [ = \n",  name);
  for (size_t i = 0; i < network->layer_count; i++) {
    mat_print(network->layers[i].ws, "ws", 4);
    mat_print(network->layers[i].ws, "bs", 4);
    mat_print(network->layers[i].ws, "as", 4);
  }
}
