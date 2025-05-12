#include <stdio.h>
#define MAT_IMPLEMENTATION
#include "neural_network.h"
#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

float sigmoidf(float x) {
  return x >= 0? (1.f / (1.f + exp(-x))) : (exp(x) / (1.f + exp(x)));
}

float reluf(float x) {
  return x > 0 ? x : x * NN_RELU_PARAM;
}

float tanhf(float x) {
  float ex = expf(x);
  float enx = expf(-x);
  return (ex - enx)/(ex + enx);
}

float actf(float x, Act act) {
  switch (act) {
    case ACT_SIG:
      return sigmoidf(x);
    case ACT_RELU:
      return reluf(x);
    case ACT_TANH:
      return tanhf(x);
    default:
      return INFINITY;
  }
}

float dactf(float y, Act act) {
  switch (act) {
    case ACT_SIG:
      return y * (1 - y);
    case ACT_RELU:
      return y >= 0 ? 1 : NN_RELU_PARAM;
    case ACT_TANH:
      return 1 - y * y;
    default:
      return INFINITY;
  }
}

float BCE(Mat predicted, Mat expected) {
  if (predicted.cols != expected.cols)
    return INFINITY;

  const float eps = 1e-7;
  float loss = 0.0f;

  for (int i = 0; i < expected.rows ; i++) {
    for (int j = 0; j < expected.cols; j++) {
      float p = MAT_AT(predicted, i, j);
      float y = MAT_AT(expected, i, j);

      if (p < eps)
        p = eps;

      if (p > 1.0f - eps) 
        p = 1.0f - eps;

      loss -= (y * logf(p)) + (1.0f - y) * logf(1.0f - p);
    }
  }
  
  return loss / expected.rows;
}

float MSE(Mat predicted, Mat expected) {
  if (predicted.cols != expected.cols)
    return INFINITY;

  float cost = 0.0f;
  for (size_t i = 0; i < predicted.rows; i++)
    for (size_t j = 0; j < predicted.cols; j++) {
      float dist = MAT_AT(predicted, i, j) - MAT_AT(expected, i, j);
      cost += dist * dist;
    }

  return cost / predicted.rows;
}

float compute_loss(Mat expected, Mat predicted, Loss lf) {
  switch (lf) {
    case LOSS_BCE:
      return BCE(predicted, expected);
    case LOSS_MSE:
      return MSE(predicted, expected);
    default:
      return INFINITY;
  }
}

float compute_loss_deriv(float activation, float expected, Loss lf) {
  switch (lf) {
    case LOSS_MSE:
      return 2 * (activation - expected);
    case LOSS_BCE:
      return activation - expected;
    default:
      return INFINITY;
  }
}

int nn_init(NN *neural_network, size_t count, size_t *layers, float rate, Loss lf) {
  if (count == 0)
    return -1;
  neural_network->loss_function = lf;
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
    mat_randomize_xavier(&current_layer->ws, layers[0], layers[count - 1]);
    mat_fill(&current_layer->bs, 0.01);
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

NN *nn_alloc(size_t count, size_t *layers, float rate, Loss lf) {
  NN *neural_network = malloc(sizeof(NN)); 
  if (!neural_network)
    return NULL;

  nn_init(neural_network, count, layers, rate, lf);
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
    // mat_activate(&network->layers[i].as, sigmoidf);
    for (size_t j = 0; j < network->layers[i].as.cols; j++)
      MAT_AT(network->layers[i].as, 0, j) = 
        actf(MAT_AT(network->layers[i].as, 0, j), NN_ACT);
  }
  return 0;
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

/*
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
        forward_propagation(neural_network);
        c1 = MSE(neural_network->layers[neural_network->layer_count - 1].as, to);

        *w = saved - eps;
        c2 = MSE(neural_network, ti, to);

        MAT_AT(grad->layers[i].ws, row, col) = (c1 - c2) / (2 * eps);
        *w = saved;
      }
    }

    for (size_t col = 0; col < neural_network->layers[i].bs.cols; col++) {
      float *w = &MAT_AT(neural_network->layers[i].bs, 0, col);
      float saved = *w;

      *w = saved + eps;
      c1 = MSE(neural_network, ti, to);

      *w = saved - eps;
      c2 = MSE(neural_network, ti, to);

      MAT_AT(grad->layers[i].bs, 0, col) = (c1 - c2) / (2 * eps);
      *w = saved;
    }
  }
}
*/

inline void static gradient_diff(NN *network, gradient * grad) {
  for (size_t i = 1; i < network->layer_count; i++) {
    mat_scalar(&grad->layers[i].ws, (-1) * network->learning_rate);  
    mat_sum(&network->layers[i].ws, &grad->layers[i].ws);           
    mat_scalar(&grad->layers[i].bs, (-1) * network->learning_rate);
    mat_sum(&network->layers[i].bs, &grad->layers[i].bs);
  }
}

int finite_diff(NN *network, gradient *grad, Mat ti, Mat to) {
  // gradient_compute(network, grad, 1e-3, ti, to);
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

  Mat x;
  Mat y_expected;
  Mat *y_predicted;

  mat_init(&x, 1, ti.cols);
  mat_init(&y_expected, 1, to.cols);

  for (size_t i = 0; i < grad->layer_count; i++) {
    mat_fill(&grad->layers[i].ws, 0.0f);
    mat_fill(&grad->layers[i].bs, 0.0f);
    mat_fill(&grad->layers[i].as, 0.0f);
  }

  for (size_t i = 0; i < ti.rows; i++) {
    // compute error for each sample

    mat_row(&x, &ti, i);
    mat_inplace_copy(&network->layers[0].as, &x);
    forward_propagation(network);

    mat_row(&y_expected, &to, i);
    y_predicted = &network->layers[network->layer_count - 1].as;

    // L - current layer
    // z(L) - pre-activation
    // b(L) - bias of current layer
    // a(L) - activation of current layer
    // w(L) - weights of current layer
    // C0 - cost of current sample

    // derivative of C0 in respect to a(L) * derivative of a(L) in respect to z(L)
    // stored inside the output layer of the gradient

    for (size_t j = 0; j < y_predicted->cols; j++) {
      float activation = MAT_AT(*y_predicted, 0, j);
      float expected = MAT_AT(y_expected, 0, j);
      float loss_deriv = compute_loss_deriv(activation, expected, network->loss_function);
      float act_deriv = dactf(activation, NN_ACT);

      MAT_AT(grad->layers[grad->layer_count - 1].as, 0, j) = 
        loss_deriv * act_deriv;
    }

    Mat *delta = &grad->layers[grad->layer_count - 1].as;

    // derivative of C0 in respect to b(L) is
    // act_deriv * error_deriv

    // derivative of C0 in respect to w(L) is
    // a(L - 1) * act_deriv * error_deriv

    // derivative of C0 in respect to a(L-1) is w(L) * act_deriv * loss_deriv

    for (size_t l = network->layer_count - 1; l > 0; --l) {
      layer *current = &network->layers[l];
      layer *previous = &network->layers[l - 1];
      layer *grad_current = &grad->layers[l];

      // derivative of b(L)
      for (size_t j = 0; j < current->bs.cols; j++) 
        MAT_AT(grad_current->bs, 0, j) += MAT_AT(*delta, 0, j);

      // derivative of w(L) 

      for (size_t j = 0; j < current->ws.rows; j++) 
        for (size_t k = 0; k < current->ws.cols; k++) {
          float d= MAT_AT(*delta, 0, k);
          float a_prev = MAT_AT(previous->as, 0, j);
          MAT_AT(grad_current->ws, j, k) += d * a_prev;
        }
      
      // derivative of a(L-1)
      if (l >= 1) {
        Mat *prev_delta = &grad->layers[l - 1].as;
        for (size_t j = 0; j < current->ws.rows; j++) {
          float sum = 0.0f;
          for (size_t k = 0; k < current->ws.cols; k++) {
            float w= MAT_AT(current->ws, j, k);
            float d = MAT_AT(*delta, 0, k);
            sum += w * d; 
          }
          float a = MAT_AT(previous->as, 0, j);
          MAT_AT(*prev_delta, 0, j) = sum * dactf(a, NN_ACT);
        }
        delta = prev_delta;
      }
    }
  }

  for (size_t l = 0; l < grad->layer_count; l++) {
    for (size_t i = 0; i < grad->layers[l].ws.rows; i++) 
      for (size_t j = 0; j < grad->layers[l].ws.cols; j++)
        MAT_AT(grad->layers[l].ws, i, j) /= ti.rows;

    for (size_t i = 0; i < grad->layers[l].bs.cols; i++)
      MAT_AT(grad->layers[l].bs, 0, i) /= ti.rows;
  }

  gradient_diff(network, grad);
  mat_deinit(&x);
  mat_deinit(&y_expected);

  return 0;
}

int learn(NN *network, Mat ti, Mat to, size_t epochs, int (*fn)(NN *network, gradient *grad, Mat ti, Mat to)) {
  gradient grad;
  if (gradient_init(&grad, network))
    return -1;

  for (size_t epoch = 0; epoch < epochs; epoch++) {
    fn(network, &grad, ti, to);
    if (epoch % 10 == 0) {
      Mat *predicted = &network->layers[network->layer_count - 1].as;
      printf("cost at epoch {%ld}: %f\n", epoch, compute_loss(to, *predicted, network->loss_function));
    }
    
  }

  gradient_deinit(&grad);

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
