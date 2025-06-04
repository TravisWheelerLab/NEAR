//
// Created by Daniel Olson on 5/29/25.
//

#include "process_hits.h"
#include "io.h"
#include "util.h"

double log_pval_for_hit(const Hit *hit, const ProcessHitArgs *args) {

  uint32_t stat_bin_start = (hit->query_bin * args->indices_per_stat_row) +
                            (hit->target_bin * args->num_distributions);

  for (uint32_t i = 0; i < args->num_distributions; ++i) {
    uint32_t stat_bin = stat_bin_start + i;

    if (hit->cosine_sim >= args->genpareto_locs[stat_bin]) {
      double log_pval = args->flat_log_additions[i];
      log_pval += genpareto_logsf(hit->cosine_sim,
                                  args->genpareto_locs[stat_bin],
                                  args->genpareto_scales[stat_bin],
                                  args->genpareto_shapes[stat_bin]);
     // printf("%i %i %i %i %i     %f %f %f %f \n", stat_bin_start, stat_bin, i, hit->query_bin, hit->target_bin,
      //log_pval, args->genpareto_locs[stat_bin], args->genpareto_scales[stat_bin], args->genpareto_shapes[stat_bin]);

      return log_pval;
    }
  }
  return 0;
}

// Filter 1 treats hits as independent
double log_pval_from_independent_hits(const ProcessHitArgs *args,
                                      const Hit *restrict hits, uint64_t start,
                                      uint64_t end, int n_rows, int n_cols) {
  /* keep only the rarest hit per target */
  uint32_t last_tid = hits[start].target_pos;
  double tid_best_logp = log_pval_for_hit(&hits[start], args);
  double logp_sum = tid_best_logp;
  int nhits = 1;

  for (uint64_t i = start + 1; i < end; ++i) {
    double hit_logp = log_pval_for_hit(&hits[i], args);
    if (hits[i].target_pos == last_tid) { /* same target */
      if (hit_logp < tid_best_logp) { /* better hit*/
        logp_sum += hit_logp - tid_best_logp;
        tid_best_logp = hit_logp;
      }
    } else { /* new target  */
      last_tid = hits[i].target_pos;
      tid_best_logp = hit_logp;
      logp_sum += tid_best_logp;
      ++nhits;
    }
  }

  double log_lambda = logp_sum + log_rook(n_rows, n_cols, nhits);
  //printf("%llu %llu %llu %f %f %f\n", nhits, n_rows, n_cols, logp_sum, log_lambda, log_poisson_tail(log_lambda));
  return log_poisson_tail(log_lambda);
}

double log_odds_transition(const ProcessHitArgs *args,
                           const Hit *first_hit,
                           const Hit *second_hit)
{
  double expected_cosine_hit = args->expected_log_cosine_dvg[first_hit->query_bin] *
                               (second_hit->query_pos - first_hit->query_pos);
  expected_cosine_hit += args->expected_log_cosine_dvg[first_hit->target_bin] *
                         (second_hit->target_pos - first_hit->target_pos);
  expected_cosine_hit = exp(expected_cosine_hit);

  Hit hit = {.query_pos=second_hit->query_pos,
             .target_pos=second_hit->target_pos,
             .query_bin=second_hit->query_bin,
             .target_bin=second_hit->target_bin,
             .cosine_sim=expected_cosine_hit};

  double log_adjustment = log_pval_for_hit(&hit, args);

  return -log_adjustment;
}

/* --------------------------------------------------------------------------
 *  Filter-2 : coherent-path p-value
 * --------------------------------------------------------------------------*/
double log_pval_from_coherent_hits(const ProcessHitArgs *args, uint64_t start,
                                   uint64_t end, uint64_t n_rows,
                                   uint64_t n_cols) {

  const Hit *restrict hits = args->hits;
  const size_t N = (size_t)(end - start);
  if (N == 0)
    return 0.0; /* empty slice → p = 1 */

  /* ---- scratch, stack-backed when N ≤ 1024 ------------------------ */

  double *dp =
      (N <= DP_STACK_LIM) ? args->dp_st : (double *)malloc(N * sizeof(*dp));
  int *plen =
      (N <= DP_STACK_LIM) ? args->ln_st : (int *)malloc(N * sizeof(*plen));

  double best_score = hits[start].cosine_sim;
  int best_len = 1;

  for (size_t i = 0; i < N; ++i) {
    const Hit *restrict hi = &hits[start + i];

    double best_i = log_pval_for_hit(hi, args); /* path that starts at i */
    int len_i = 1;

    /* ---------- inner scan, backwards, branch-light ------------- */
    for (ssize_t j = (ssize_t)i - 1; j >= 0; --j) {
      const Hit *restrict hj = &hits[start + j];

      /* cheap rejection first */
      if (hj->target_pos == hi->target_pos)
        continue; /* same col */
      if (hj->query_pos >= hi->query_pos)
        continue; /* wrong col */

      /* passed the two filters -> valid predecessor */
      double trans = log_odds_transition(args, hi, hj);

      double cand = dp[j] + trans + log_pval_for_hit(hi, args);
      if (cand < best_i) { /* smaller -> rarer -> better -> faster ->stronger*/
        best_i = cand;
        len_i = plen[j] + 1;
      }
    }

    dp[i] = best_i;
    plen[i] = len_i;

    if (best_i < best_score) {
      best_score = best_i;
      best_len = len_i;
    }
  }

  /* convert path score to Poisson tail, same as Filter-1 */
  double log_lambda = best_score + log_rook((int)n_rows, (int)n_cols, best_len);
  double log_pval = log_poisson_tail(log_lambda);

  if (dp != args->dp_st)
    free(dp);
  if (plen != args->ln_st)
    free(plen);

  return log_pval; /* already log(p-value) */
}

void process_hit_range(const ProcessHitArgs *args, uint64_t starting_index,
                       uint64_t ending_index) {
  QueryTargetSimilarity qt_sim;

  uint64_t query_length =
      args->query_lengths[args->hits[starting_index].query_seq_id];
  uint64_t target_length =
      args->target_lengths[args->hits[starting_index].target_seq_id];

  // Calculate first filter pval
  qt_sim.log_pval_filter_1 = log_pval_from_independent_hits(args,
                                                            args->hits,
                                                            starting_index,
                                                            ending_index,
                                                            query_length,
                                                            target_length*args->sparsity);
  if (qt_sim.log_pval_filter_1 < args->filter_1_logpval_threshold) {
    // If it passes first filter, calculate second filter pval
    qt_sim.log_pval_filter_2 = log_pval_from_coherent_hits(
        args, starting_index, ending_index, query_length, target_length*args->sparsity);
    if (qt_sim.log_pval_filter_2 < args->filter_2_logpval_threshold) {
      // Output the query target pair it passes the second filter
      qt_sim.query_seq_id = args->hits[starting_index].query_seq_id;
      qt_sim.target_seq_id = args->hits[starting_index].target_seq_id;
      output_similarity(args, qt_sim);
    }
  }
}

void process_hits(ProcessHitArgs args) {

  const Hit *hits = args.hits;

  uint32_t last_query = hits[0].query_seq_id;
  uint32_t last_target = hits[0].target_seq_id;

  uint64_t starting_index = 0;

  for (uint64_t i = 1; i < args.num_hits; ++i) {
    uint32_t q = hits[i].query_seq_id;
    uint32_t t = hits[i].target_seq_id;

    if (q != last_query || t != last_target) {
      process_hit_range(&args, starting_index, i);

      starting_index = i;
      last_query = q;
      last_target = t;
    }
  }
  // final flush
  if (last_query != (uint32_t)-1) {
    process_hit_range(&args, starting_index, args.num_hits);
  }
}