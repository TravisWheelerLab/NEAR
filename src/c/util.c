//
// Created by Daniel Olson on 5/29/25.
//

#include "util.h"
#include "types.h"

double log1mexp(double x) {
  if (x >= 0.0) {
    // log(1 - exp(x)) is undefined for x >= 0
    return -INFINITY;
  }
  if (x < LOG_HALF) {
    return log1p(-exp(x));     // safe when exp(x) is small
  } else {
    return log(-expm1(x));     // safe when exp(x) ≈ 1
  }
}

double log_rook(double a, double b, double k) {
  return lgamma(a + 1.0) - lgamma(a - k + 1.0) +
         lgamma(b + 1.0) - lgamma(b - k + 1.0) -
         (2 * lgamma(k + 1.0));
}

double log_ch(double a, double k) {
  // printf("%f %f %f\n", lgamma(a + 1.0), -lgamma(a - k + 1.0), -(1 * lgamma(k + 1.0)));
  return lgamma(a + 1.0) - (lgamma(a - k + 1.0) + (1 * lgamma(k + 1.0)));
}

double log_sum_binom(double N) {
    const double LOG2 = 0.6931471805599453; // precomputed log(2)
    double x = N * LOG2;
    return x + log1mexp(x);
}

double log_poisson_tail(double log_lambda) {
  if (log_lambda <= LOG_LAM_SMALL) /* tiny lambda */
    return log_lambda;             /* log p ~ log lambda        */

  if (log_lambda >= LOG_LAM_LARGE) /* huge lambda */
    return 0.0;                    /* p ~ 1, log p ~ 0     */

  double lambda = exp(log_lambda); /* safe: lambda < e^50 */
  return log1mexp(-lambda);
}

uint64_t seqlist_size(const uint64_t *seq_lengths, uint64_t num_lengths) {

  uint64_t total_embeddings = 0;
  for (uint64_t i = 0; i < num_lengths; ++i) {
    total_embeddings += seq_lengths[i];
  }

  return total_embeddings;
}
