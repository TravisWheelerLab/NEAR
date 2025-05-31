//
// Created by Daniel Olson on 5/29/25.
//

#ifndef PROCESS_HITS_PROCESS_HITS_H
#define PROCESS_HITS_PROCESS_HITS_H
#include "types.h"

// Filter 1 treats hits as independent
double log_pval_from_independent_hits(const Hit *hits, uint64_t start,
                                      uint64_t end, int n_rows, int n_cols);

double log_odds_transition(uint32_t q_i, uint32_t t_i, uint32_t q_j,
                           uint32_t t_j, double logp_i, double logp_j);

double log_pval_from_coherent_hits(const ProcessHitArgs *args, uint64_t start,
                                   uint64_t end, uint64_t n_rows,
                                   uint64_t n_cols);

void process_hit_range(const ProcessHitArgs *args, uint64_t starting_index,
                       uint64_t ending_index);

void process_hits(ProcessHitArgs args);

#endif // PROCESS_HITS_PROCESS_HITS_H
