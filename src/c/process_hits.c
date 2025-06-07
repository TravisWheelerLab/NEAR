//
// Created by Daniel Olson on 5/29/25.
//

#include "process_hits.h"
#include "io.h"
#include "util.h"

double log_pval_for_hit(const Hit *hit, const ProcessHitArgs *args) {
  double def = 0;//-log(1e-3);
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

      return log_pval + def;// - LOG_HALF;
    }
  }
  return def;// + def;
}

double excluded_area_for_start(double start_q, double start_t, double q_len, double t_len) {
  double total_area;
  start_q += 1;
  start_t += 1;
  q_len += 1;
  t_len += 1;
  total_area = total_area + (start_q * t_len);
  total_area = total_area + (start_t * q_len);
  total_area = total_area - (start_q * start_t);
  return total_area;
}

// Filter 1 treats hits as independent
double log_pval_from_independent_hits(const ProcessHitArgs *args,
                                      const Hit *hits, uint64_t start,
                                      uint64_t end, double query_length,
                                      double target_length,
                                      uint32_t *nhits) {
  /* keep only the rarest hit per target */
  uint32_t last_tid = hits[start].target_pos;
  double tid_best_logp = log_pval_for_hit(&hits[start], args);
  double logp_sum = tid_best_logp;
  int num_hits = 1;

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
      ++num_hits;
    }
  }

  if (num_hits > target_length) {
    num_hits = target_length;
  }
  if (num_hits > query_length) {
    num_hits = query_length;
  }

  double log_arg = logp_sum + log(query_length * target_length);
  double log_pval = log_poisson_tail(log_arg);

  if (nhits != NULL)
    *nhits = num_hits;
  return log_pval;
}

double logp_hit_given_hit(const ProcessHitArgs *args,
                           const Hit *first_hit,
                           const Hit *second_hit)
{
  double log_theta_q = args->expected_log_cosine_dvg[first_hit->query_bin];
  double log_theta_t = args->expected_log_cosine_dvg[first_hit->target_bin];

  double dist_q = second_hit->query_pos - first_hit->query_pos;
  double dist_t = second_hit->target_pos - first_hit->target_pos;

  double expected_cosine_dvg = (log_theta_q * dist_q) + (log_theta_t * dist_t);
  expected_cosine_dvg = exp(expected_cosine_dvg);

  Hit hit = {.query_pos=second_hit->query_pos,
             .target_pos=second_hit->target_pos,
             .query_bin=second_hit->query_bin,
             .target_bin=second_hit->target_bin,
             .cosine_sim=expected_cosine_dvg * first_hit->cosine_sim};

  return log_pval_for_hit(&hit, args);
}

/* --------------------------------------------------------------------------
 *  Filter-2 : coherent-path p-value
 * --------------------------------------------------------------------------*/
double log_pval_from_coherent_hits(const ProcessHitArgs *args, uint64_t start,
                                   uint64_t end, double query_length,
                                   double target_length, uint32_t *nhits) {

  const Hit *hits = args->hits;
  const size_t N = (size_t)(end - start);
  double inv_sparsity = 1.0 / args->sparsity;

  if (N == 0)
    return 0.0; /* empty slice → p = 1 */

  double *dp =
      (N <= DP_STACK_LIM) ? args->dp_st : (double *)malloc(N * sizeof(*dp));
  float *plen =
      (N <= DP_STACK_LIM) ? args->ln_st : (float *)malloc(N * sizeof(float));


  volatile double combined_score = 1000.0; // big bug if not volatile
  volatile double remaining_area = 0;
  double best_len = 1;


  for (size_t i = 0; i < N; ++i) {
    const Hit *hi = &hits[start + i];

    double hi_hit_p = log_pval_for_hit(hi, args);
    double excluded_area = excluded_area_for_start(hi->query_pos,
                                                   hi->target_pos,
                                                   query_length,
                                                   target_length * inv_sparsity);


    double best_i = hi_hit_p + log(excluded_area); /* path that starts at i */
    float len_i = 1;

    /* ---------- inner scan, backwards, branch-light ------------- */
    for (ssize_t j = (ssize_t)i - 1; j >= 0; --j) {
      const Hit *hj = &hits[start + j];

      /* cheap rejection first */
      if (hj->target_pos == hi->target_pos)
        continue; /* same col */
      if (hj->query_pos >= hi->query_pos)
        continue; /* wrong col */

      /* passed the two filters -> valid predecessor  */
      double hj_excluded_area = excluded_area - excluded_area_for_start(hj->query_pos,
                                                                        hj->target_pos,
                                                                        query_length,
                                                                        target_length * inv_sparsity);

      double conditional_hit_p = logp_hit_given_hit(args, hj, hi);
      double cand = dp[j] + conditional_hit_p + log(hj_excluded_area);
      //  printf("%f %f %f %f %f\n", cand, dp[j], hi_hit_p, log(hj_excluded_area), hj_excluded_area);

      if (cand < best_i) { /* smaller -> rarer -> better -> faster ->stronger*/
        best_i = cand;
        len_i = plen[j] + 1;
      }
    }

    dp[i] = best_i;
    plen[i] = len_i;
   // printf("Checking if best_i better than combined score: %f %p\n", combined_score, (void*)&combined_score);
    if (best_i < combined_score) {
      combined_score = best_i;
      best_len = len_i;
      remaining_area = (query_length - hi->query_pos + 1) * ((target_length * inv_sparsity) - hi->target_pos + 1);
    }
  }

  double log_pval = log_poisson_tail(combined_score +
                                     log(remaining_area));
  if (dp != args->dp_st)
    free(dp);
  if (plen != args->ln_st)
    free(plen);
  if (nhits != NULL)
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
    qt_sim.log_pval_filter_2 = log_pval_from_coherent_hits(args,
                                                           starting_index,
                                                           ending_index, query_length,
                                                           target_length,
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