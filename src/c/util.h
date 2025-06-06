//
// Created by Daniel Olson on 5/29/25.
//

#ifndef PROCESS_HITS_UTIL_H
#define PROCESS_HITS_UTIL_H
#include <stdint.h>
#include <math.h>
#include "types.h"

double log_ch(double a, double k);
double log1mexp(double x);
double log_rook(double a, double b, double k);
double log_poisson_tail(double log_lambda);
double log_sum_binom(double N);

static inline double genpareto_logsf(double x, double loc, double scale,
                                     double shape) {
  x = (x - loc) / scale;
  x = 1 + (x * shape);
  if (x <= 0)
    return -500.0;
  x = log(x) * (-1.0 / shape);
  return x; // subtract ln(0.5)
}

uint64_t seqlist_size(const uint64_t *seq_lengths, uint64_t num_lengths);

#endif // PROCESS_HITS_UTIL_H
