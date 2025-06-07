//
// Created by Daniel Olson on 5/29/25.
//

#ifndef PROCESS_HITS_PROCESS_HITS_H
#define PROCESS_HITS_PROCESS_HITS_H
#include "types.h"

double log_filter1_sum_t_pval(double filter1_sum, double n, double m, double hits);

double log_pval_for_hit(const Hit *hit, const ProcessHitArgs *args);

// Filter 1 treats hits as independent
double log_pval_from_independent_hits(const ProcessHitArgs *args,
                                      const Hit *hits, uint64_t start,
                                      uint64_t end, double query_length,
                                      double target_length, uint32_t*nhits);

double logp_hit_given_hit(const ProcessHitArgs *args,
                           const Hit *first_hit,
                           const Hit *second_hit);

double excluded_area_for_start(double start_q, double start_t, double q_len, double t_len);

double log_pval_from_coherent_hits(const ProcessHitArgs *args, uint64_t start,
                                   uint64_t end, double query_length,
                                   double target_length, uint32_t *nhits);

void process_hit_range(const ProcessHitArgs *args, uint64_t starting_index,
                       uint64_t ending_index);

void process_hits(ProcessHitArgs args);

#endif // PROCESS_HITS_PROCESS_HITS_H
