//
// Created by Daniel Olson on 5/29/25.
//

#ifndef PROCESS_HITS_TYPES_H
#define PROCESS_HITS_TYPES_H

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define OUTPUT_BUF_SIZE (1 << 25)
#define TID_TO_SEQID(x) (x >> 32)
#define DP_STACK_LIM 1048576
#define TID_TO_BIN(x) (x & 0x7F)
#define TID_TO_POS(x) ((x & 0xFFFFFFFF) >> 7)

#define LOG_LAM_SMALL (-10.0)
#define LOG_LAM_LARGE 50.0
#define LOG_HALF (-0.6931471805599453)

typedef struct {
  uint32_t query_seq_id;
  uint32_t target_seq_id;

  uint32_t query_pos;
  uint32_t target_pos;

  uint8_t query_bin;
  uint8_t target_bin;

  double cosine_sim;
} Hit;

typedef struct {
  uint32_t query_seq_id;
  uint32_t target_seq_id;

  uint32_t num_unique_hits;
  uint32_t coherent_length;

  double log_pval_filter_1;
  double log_pval_filter_2;
} QueryTargetSimilarity;

typedef struct {
  FILE *out;

  double filter_1_logpval_threshold;
  double filter_2_logpval_threshold;

  uint64_t num_hits;
  const Hit *hits;

  double *dp_st;
  int *ln_st;

  uint64_t num_query_seqs;
  const uint64_t *query_name_starts;
  const char *query_names;

  uint64_t num_target_seqs;
  const uint64_t *target_name_starts;
  const char *target_names;

  const uint64_t *query_lengths;
  const uint64_t *target_lengths;

  uint64_t index_size;
  uint32_t hits_per_emb;

  double sparsity;

  int n_threads;
  int thread_id;
  int num_stat_bins;

  double flat_log_addition;

  // Each of these is expected to be num_stat_bins X num_stat_bins
  double *genpareto_locs;
  double *genpareto_scales;
  double *genpareto_shapes;

  // Expected to be num_stat_bins
  double *expected_log_cosine_dvg;

} ProcessHitArgs;

static inline int cmp_hit(const void *a, const void *b) {
  const Hit *pa = a;
  const Hit *pb = b;

  if (pa->query_seq_id < pb->query_seq_id)
    return -1;
  if (pa->query_seq_id > pb->query_seq_id)
    return 1;

  if (pa->target_seq_id < pb->target_seq_id)
    return -1;
  if (pa->target_seq_id > pb->target_seq_id)
    return 1;

  if (pa->target_pos < pb->target_pos)
    return -1;
  if (pa->target_pos > pb->target_pos)
    return 1;

  return 0;
}

#endif // PROCESS_HITS_TYPES_H
