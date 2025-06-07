//
// Created by Daniel Olson on 5/29/25.
//

#include "process_hits.h"
#include "io.h"
#include "util.h"

double log_pval_for_hit(const Hit *hit, const ProcessHitArgs *args) {
  double def = -log(1e-3);
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

      return log_pval + def;// - LOG_HALF;
    }
  }
  return def;// + def;
}

// Filter 1 treats hits as independent
double log_pval_from_independent_hits(const ProcessHitArgs *args,
                                      const Hit *restrict hits, uint64_t start,
                                      uint64_t end, int n_rows, int n_cols,
                                      int *nhitsp) {
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

  if (nhits > n_cols) {
    nhits = n_cols;
  }
  if (nhits > n_rows) {
    nhits = n_rows;
  }

  double logpval_adjust = lgamma(n_cols + n_rows);

  double h = nhits;
  double c = n_cols;
  double r = n_rows;

  double score_adjust = log_ch(r, h) + log_ch(c, h);

  //double hit_prob = (double)args->hits_per_emb / (double)args->index_size;
  score_adjust = log(r * c);
  double log_arg = logp_sum + score_adjust;

  double log_pval = log_poisson_tail(log_arg);
  return log_pval;
}

double log_odds_transition(const ProcessHitArgs *args,
                           const Hit *first_hit,
                           const Hit *second_hit,
                           float *k)
{



  double log_theta_q = args->expected_log_cosine_dvg[first_hit->query_bin];
  double log_theta_t = args->expected_log_cosine_dvg[first_hit->target_bin];

  double dist_q = second_hit->query_pos - first_hit->query_pos;
  double dist_t = second_hit->target_pos - first_hit->target_pos;

  double expected_cosine_dvg = (log_theta_q * dist_q) + (log_theta_t * dist_t);
  expected_cosine_dvg = exp(expected_cosine_dvg);

  if (fabs(dist_q - dist_t) != 0) {
    return -(log(0.02) + (log(0.1) * MIN(5.0, fabs(dist_q - dist_t - 1.0)))); // gap penalty
  }

  *k += (1.0 - expected_cosine_dvg);
  Hit hit = {.query_pos=second_hit->query_pos,
             .target_pos=second_hit->target_pos,
             .query_bin=second_hit->query_bin,
             .target_bin=second_hit->target_bin,
             .cosine_sim=expected_cosine_dvg * first_hit->cosine_sim};

  double log_adjustment = (log_pval_for_hit(&hit, args) * expected_cosine_dvg);

  if (fabs(dist_q - dist_t) != 0) {
    log_adjustment += log(0.02) + (log(0.1) * MIN(5.0, fabs(dist_q - dist_t - 1.0))); // gap penalty
  }

  log_adjustment += -log(dist_q*dist_t)*0.5; // sparsity penalty

  return -log_adjustment;
}

/* --------------------------------------------------------------------------
 *  Filter-2 : coherent-path p-value
 * --------------------------------------------------------------------------*/
double log_pval_from_coherent_hits(const ProcessHitArgs *args, uint64_t start,
                                   uint64_t end, uint64_t n_rows,
                                   uint64_t n_cols, int*nhits) {

  const Hit *restrict hits = args->hits;
  const size_t N = (size_t)(end - start);
  if (N == 0)
    return 0.0; /* empty slice → p = 1 */

  double *dp =
      (N <= DP_STACK_LIM) ? args->dp_st : (double *)malloc(N * sizeof(*dp));
  int *tplen =
      (N <= DP_STACK_LIM) ? args->ln_st : (int *)malloc(N * sizeof(int));

  float *plen = (float *)tplen;
  double best_score = 1;
  double best_len = 1;
  double best_sub_len = 1;

  double START_PENALTY = 0;
  //START_PENALTY += log((double)args->hits_per_emb / (double)args->index_size) * 0.5;
  double best_effective_len = 0;
  double best_effective_score = 0;

  for (size_t i = 0; i < N; ++i) {
    const Hit *restrict hi = &hits[start + i];

    double hi_hit_p = log_pval_for_hit(hi, args);
    double best_i = hi_hit_p + START_PENALTY; /* path that starts at i */
    float len_i = 1;

    /* ---------- inner scan, backwards, branch-light ------------- */
    for (ssize_t j = (ssize_t)i - 1; j >= 0; --j) {
      const Hit *restrict hj = &hits[start + j];

      /* cheap rejection first */
      if (hj->target_pos == hi->target_pos)
        continue; /* same col */
      if (hj->query_pos >= hi->query_pos)
        continue; /* wrong col */

      /* passed the two filters -> valid predecessor  */
      float ki = 0;
      double trans = log_odds_transition(args, hj, hi, &ki);
      double cand = dp[j] + trans + hi_hit_p;
      if (cand < best_i) { /* smaller -> rarer -> better -> faster ->stronger*/
        best_i = cand;
        len_i = plen[j] + 1;
      }
    }

    dp[i] = best_i;
    plen[i] = len_i;

    if (best_i < best_effective_score) {
      best_effective_score = best_i;
      best_effective_len = len_i;
    }
  }
  best_len = best_effective_len;
  double query_length = args->query_lengths[hits[start].query_seq_id];
  double target_length = args->target_lengths[hits[start].target_seq_id];
  double hit_prob = (double)args->hits_per_emb / (double)args->index_size;
  double lambda = 1.0;
  double K = 2.3;
  //printf("%f\n", best_effective_score);
  double log_pval = best_effective_score + log(query_length * target_length / (double)args->sparsity);
   /*+ log(query_length * target_length);/*(best_effective_score * lambda) + log(K) +
                    log(target_length * ((double)args->hits_per_emb) / ((double)args->index_size)) +
                    log(query_length); */
  printf("%f: %f %f %f %f ", best_effective_score, log_pval, query_length, target_length, best_effective_len);
  printf(": %f %f %f %f %f\n", log(hit_prob),
  log_ch(target_length, best_len),
  log_ch(query_length, best_len),
  log(query_length*target_length), best_len);
 // log_pval = log_poisson_tail(log_pval);
  if (dp != args->dp_st)
    free(dp);
  if (plen != args->ln_st)
    free(plen);
  *nhits = best_len;

  return log_pval;
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
                                                            target_length,
                                                            &qt_sim.num_unique_hits);

  if (qt_sim.log_pval_filter_1 <= args->filter_1_logpval_threshold) {
    // If it passes first filter, calculate second filter pval
    qt_sim.log_pval_filter_2 = log_pval_from_coherent_hits(
        args, starting_index, ending_index, query_length, target_length,
        &qt_sim.coherent_length);
    if (qt_sim.log_pval_filter_2 <= args->filter_2_logpval_threshold) {
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