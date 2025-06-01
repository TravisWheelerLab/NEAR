//
// Created by Daniel Olson on 5/29/25.
//

#ifndef PROCESS_HITS_UTIL_H
#define PROCESS_HITS_UTIL_H
#include <stdint.h>
#include <math.h>

double log1mexp(double x);
double log_rook(double a, double b, double k);
double log_poisson_tail(double log_lambda);

static inline double genpareto_logsf(double x, double loc, double scale,
                                     double shape) {
  x = (x - loc) / scale;
  x = log(1 + (x * shape)) * (-1.0 / shape);
  return x;
}

uint64_t seqlist_size(const uint64_t *seq_lengths, uint64_t num_lengths);

#endif // PROCESS_HITS_UTIL_H
