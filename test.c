#include <stdio.h>
#include "neural_network.h"

float td[] = {
  0, 0, 0,
  0, 1, 1,
  1, 0, 1,
  1, 1, 0,
};

int main() {
  Mat m;
  // test for init and randomize
  mat_init(&m, 3, 3);
  mat_randomize_xavier(&m, 0, 1);

  // test for mat_row
  Mat row = {0};
  mat_row(&row, &m, 1);
  mat_row(&row, &m, 2);

  // test for mat_deep_copy
  Mat copy = {0};
  mat_deep_copy(&copy, &m);

  // test for mat_inplace_copy
  Mat cprow = {0};
  mat_row(&cprow, &copy, 0);
  mat_inplace_copy(&cprow, &row);

  // test for mat_fill
  mat_fill(&copy, 0.1f);

  // test for mat_scalar
  mat_scalar(&copy, 10);

  // test for mat_sum
  mat_sum(&copy, &m);

  // test for mat_multiply
  Mat result;
  if (mat_init(&result, m.rows, m.cols))
    perror("error init");
  if (mat_multiply(&result, &copy, &m))
    perror("error");


  NN neural_network;
  size_t layers[] = {2, 2, 1};
  nn_init(&neural_network, 3, layers, 0.1, ACT_TANH, LOSS_MSE);

  size_t stride = 3;
  size_t sample_size = sizeof(td) / sizeof(td[0]) / stride;
  Mat ti = {
    .rows = sample_size,
    .cols = 2,
    .stride = 3,
    .es = td,
  };
  Mat to = {
    .rows = sample_size,
    .cols = 1,
    .stride = 3,
    .es = &td[2],
  };

  learn(&neural_network, ti, to, 7500, backward_propagation);

  Mat input;
  mat_init(&input, 1, ti.cols);
  float threshold = 0.5f;

  for (size_t i = 0; i < 4; i++) {
      mat_row(&input, &ti, i);
      mat_inplace_copy(&neural_network.layers[0].as, &input);
      forward_propagation(&neural_network);
      float computed_y = MAT_AT(neural_network.layers[neural_network.layer_count - 1].as, 0, 0); 
      printf("y_computed: %f ", computed_y);
      if (computed_y < threshold) {
        printf("%zu & %zu = %d\n", (size_t) MAT_AT(input, 0, 0), (size_t) MAT_AT(input, 0, 1), 0);
        continue;
      }	
      printf("%zu & %zu = %d\n", (size_t) MAT_AT(input, 0, 0) , (size_t) MAT_AT(input, 0, 1), 1);
  }

  nn_deinit(&neural_network);
  mat_deinit(&input);
  mat_deinit(&m);
  mat_deinit(&copy);
  mat_deinit(&cprow);
  mat_deinit(&row);
  mat_deinit(&result);
  return 0;
}
