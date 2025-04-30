#ifndef MAT_H_
#define MAT_H_

#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifndef MAT_MALLOC 
#define MAT_MALLOC malloc
#endif // MAT_MALLOC

#ifndef MAT_FREE
#define MAT_FREE free
#endif // MAT_FREE

#ifndef MAT_ASSERT
#define MAT_ASSERT assert
#endif // MAT_ASSERT

/**
 * @brief Prints a matrix with its name
 * @param m The matrix to print
 */
#define MAT_PRINT(m) mat_print(m, #m, 0)

/**
 * @brief Access a matrix element at position (i, j)
 * @param m The matrix
 * @param i Row index
 * @param j Column index
 * @return The element at position (i, j)
 */
#define MAT_AT(m, i, j) (m).es[(i)*(m).stride + (j)]

/**
 * @brief Function pointer type for activation functions
 * @param x The input value
 * @return The activated value
 */
typedef float (*ActivationFunc)(float x);

typedef struct {
  size_t rows;
  size_t cols;
  size_t stride;
  float *es;
} Mat;

/**
 * @brief Initialize a matrix with given dimensions
 * @param m Pointer to the matrix to initialize
 * @param rows Number of rows
 * @param cols Number of columns
 * @return 0 on success, -1 on failure
 */
int mat_init(Mat * const m, size_t rows, size_t cols);

/**
 * @brief Free resources associated with a matrix
 * @param m Pointer to the matrix to deinitialize
 * @return 0 on success, -1 on failure
 */
int mat_deinit(Mat * const m);

/**
 * @brief Create a new matrix with given dimensions
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Pointer to the created matrix or NULL on failure
 */
Mat *mat_create(size_t rows, size_t cols); 

/**
 * @brief Destroy a matrix created with mat_create
 * @param m Pointer to the matrix to destroy
 * @return 0 on success, -1 on failure
 */
int mat_destroy(Mat *m);


/**
 * @brief Copies a row from a matrix
 * @param dst Destination matrix (should be 1xN)
 * @param src Source matrix
 * @param row Row index to extract
 * @return 0 on success, -1 on failure
 */
int mat_row(Mat* dst, const Mat * const src, size_t row);

/**
 * @brief Extract a row from a matrix
 * @param src Source matrix
 * @param row Row index to extract
 * @return Mat on success, NULL on failure
 */

Mat extract_row(const Mat * const src, size_t row); 

/**
 * @brief Extract a column from a matrix
 * @param dst Destination matrix (should be Mx1)
 * @param src Source matrix
 * @param col Column index to extract
 * @return 0 on success, -1 on failure
 */
int mat_col(Mat* dst, const Mat * const src, size_t col);

/**
 * @brief Extract a column from a matrix
 * @param src Source matrix
 * @param col Column index to extract
 * @return Mat on success, NULL on failure
 */

Mat extract_col(const Mat * const src, size_t row);

/**
 * @brief Copy a matrix into another existing one
 * @param dst Destination matrix (must have same dimensions as src)
 * @param src Source matrix
 * @return 0 on success, -1 on failure
 */
int mat_inplace_copy(Mat *dst, const Mat * const src);

/**
 * @brief Create a new copy of a matrix
 * @param dst Destination matrix (will be resized if necessary)
 * @param src Source matrix
 * @return 0 on success, -1 on failure
 */
int mat_deep_copy(Mat *dst, const Mat * const src);

/**
 * @brief Fill a matrix with random values within a given range
 * @param m Matrix to randomize
 * @param lower_bound Lower bound for random values
 * @param upper_bound Upper bound for random values
 * @return 0 on success, -1 on failure
 */
int mat_randomize(Mat *m, float lower_bound, float upper_bound);

/**
 * @brief Fill a matrix with a constant value
 * @param m Matrix to fill
 * @param value Value to fill the matrix with
 * @return 0 on success, -1 on failure
 */
int mat_fill(Mat * const m, float value);

/**
 * @brief Print a matrix with a given name
 * @param m Matrix to print
 * @param name Name to display before the matrix
 */
void mat_print(const Mat m, char *name, size_t padding);

/**
 * @brief Multiply two matrices
 * @param result Result matrix (must have proper dimensions)
 * @param left Left operand
 * @param right Right operand
 * @return 0 on success, -1 on failure
 * @note result must have dimensions left->rows x right->cols
 */
int mat_multiply(Mat * const result, const Mat * const left, const Mat * const right);

/**
 * @brief Multiply all elements of a matrix by a scalar
 * @param m Matrix to scale
 * @param value Scalar value
 * @return 0 on success, -1 on failure
 */
int mat_scalar(Mat * const m, float value);

/**
 * @brief Add another matrix to the destination matrix
 * @param dst Destination matrix (will be modified)
 * @param a Matrix to add
 * @return 0 on success, -1 on failure
 */
int mat_sum(Mat * const dst, const Mat * const a);

/**
 * @brief Apply an activation function to all elements of a matrix
 * @param m Matrix to activate
 * @param function Activation function to apply
 */
void mat_activate(Mat * const m, ActivationFunc function);

#endif // MAT_H_

#ifdef MAT_IMPLEMENTATION

int mat_init(Mat * const m, size_t rows, size_t cols) {
  if (rows == 0 || cols == 0)
    return -1;
  m->rows = rows;
  m->cols = cols;
  m->stride = cols; // for the moment
  m->es = calloc((m->rows * m->cols + 1), sizeof(*m->es));

  if (!m->es)
    return -1;

  return 0;
}

int mat_deinit(Mat * const m) {
  if (m->es)
    MAT_FREE(m->es);
  m->es = 0;
  m->rows = 0;
  m->cols = 0;
  m->stride = 0;
  return 0;
}

int mat_destroy(Mat *m) {
  mat_deinit(m);
  free(m);
  m = 0;
  return 0;
}

Mat *mat_create(size_t rows, size_t cols) {
  if (rows == 0 || cols == 0)
    return NULL;
  Mat *mat = malloc(sizeof(Mat));
  if (!mat)
    return NULL;

  if (mat_init(mat, rows, cols)) {
    mat_destroy(mat);
    return NULL;
  }

  return mat;
}

int mat_row(Mat *dst, const Mat * const src, size_t row) {
  if (row >= src->rows)
    return -1;
  if (dst->cols != src->cols)
    return -1;
    
  memcpy(dst->es, &MAT_AT(*src, row, 0), dst->cols * sizeof(*dst->es));
  return 0;
}

Mat extract_row(const Mat * const src, size_t row) {
  Mat mat = {
    .rows = 1,
    .cols = src->cols,
    .stride = src->stride,
    .es = &MAT_AT(*src, row, 0)
  };
  return mat;
}

int mat_col(Mat *dst, const Mat * const src, size_t col) {
  if (mat_deinit(dst)) // in case there are some leftovers
    return -1;
  if (mat_init(dst, src->rows, 1))
    return -1;
  
  for (size_t i = 0; i < src->rows; i++) {
    MAT_AT(*dst, i, 0) =  MAT_AT(*src, i, col);
  }
  return 0;
}

Mat extract_col(const Mat * const src, size_t col) { 
  Mat mat; 
  mat_init(&mat, src->rows, 1);
  for (size_t i = 0; i < src->rows; i++) {
    MAT_AT(mat, i, 0) =  MAT_AT(*src, i, col);
  }

  return mat;
}

void mat_print(Mat m, char *mname, size_t padding) {
  printf("%*s%s [ = \n", (int) padding, "", mname);
  for (size_t i = 0; i < m.rows; i++) {
    printf("%*s", (int) padding, "");
    for (size_t j = 0; j < m.cols; j++)
      printf("%f, ", MAT_AT(m, i, j));
    printf("%*s\n", (int) padding, "");
  }
  printf("%*s]\n", (int) padding, "");
}

static inline float rand_float(void) {
	return (float) rand() / (float ) RAND_MAX;
}

int mat_randomize(Mat *m, float lower_bound, float upper_bound) {
  if (lower_bound == upper_bound)
    return -1;
  static int seeded = 0;
  if (!seeded) {
    srand((unsigned)time(NULL));
    seeded = 1;
  }

  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++)
      MAT_AT(*m, i, j) = rand_float() * ( upper_bound - lower_bound) + lower_bound;
  }
  return 0;
}

int mat_inplace_copy(Mat *dst, const Mat * const src) {
  if (src->rows != dst->rows)
    return -1;

  if (src->cols != dst->cols)
    return -1;

  memcpy(dst->es, src->es, sizeof(*src->es) * src->rows * src->cols);
  return 0;
}

int mat_deep_copy(Mat *dst, const Mat * const src) {
  if (mat_deinit(dst))
    return -1;

  if (mat_init(dst, src->rows, src->cols))
    return -1;

  memcpy(dst->es, src->es, sizeof(*src->es) * src->rows * src->cols);
  return 0;
}

int mat_fill(Mat * const m, float value) {
    for (size_t i = 0; i < m->rows; i++)
      for (size_t j = 0; j < m->cols; j++)
        MAT_AT(*m, i, j) = value;
  return 0;
}

int mat_multiply(Mat * const result, const Mat * const left, const Mat * const right) {
  if (left->cols != right->rows)
    return -1;

  if (result->rows != left->rows || result->cols != right->cols)
    return -1;

  for (size_t i = 0; i < left->rows; i++)
    for (size_t j = 0; j < right->cols; j++) {
      for (size_t k = 0; k < left->cols; k++)
        MAT_AT(*result, i, j) += MAT_AT(*left, i, k) * MAT_AT(*right, k, j);
    }    
  return 0;
}

int mat_scalar(Mat * const m, float value) {
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++)
      MAT_AT(*m, i, j) *= value;
  }
  return 0;
}

int mat_sum(Mat * const dst, const Mat * const a) {
  if (dst->rows != a->rows || dst->cols != a->cols)
    return -1;

  for (size_t i = 0; i < dst->rows; i++) {
    for (size_t j = 0; j < dst->cols; j++)
      MAT_AT(*dst, i, j) += MAT_AT(*a, i, j);
  }
  return 0;
}

void mat_activate(Mat * const m, ActivationFunc function) {
  for (size_t i = 0; i < m->rows; i++) {
    for (size_t j = 0; j < m->cols; j++)
      MAT_AT(*m, i, j) = function(MAT_AT(*m, i, j));
  }
}

#endif // MAT_IMPLEMENTATION_

