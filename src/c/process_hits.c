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
  return (q_len - start_q) * (t_len - start_t);
}

// Filter 1 treats hits as independent
double log_pval_from_independent_hits(const ProcessHitArgs *args,
                                      const Hit *hits, uint64_t start,
                                      uint64_t end, double query_length,
                                      double target_length,
                                      uint32_t *nhits) {
  /* keep only the rarest hit per target */
  uint32_t last_tid = hits[start].target_seq_pos;
  double tid_best_logp = log_pval_for_hit(&hits[start], args);
  double logp_sum = tid_best_logp;
  int num_hits = 1;

  for (uint64_t i = start + 1; i < end; ++i) {
    double hit_logp = log_pval_for_hit(&hits[i], args);
    if (hits[i].target_seq_pos == last_tid) { /* same target */
      if (hit_logp < tid_best_logp) { /* better hit*/
        logp_sum += hit_logp - tid_best_logp;
        tid_best_logp = hit_logp;
      }
    } else { /* new target  */
      last_tid = hits[i].target_seq_pos;
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

double expected_hit_logp(const ProcessHitArgs *args,
                           const Hit *first_hit,
                           const Hit *second_hit,
                         double *expected_cosine_dvg)
{
  *expected_cosine_dvg = 0;

  double log_theta_q = args->expected_log_cosine_dvg[first_hit->query_bin];
  double log_theta_t = args->expected_log_cosine_dvg[first_hit->target_bin];

  double dist_q = second_hit->query_seq_pos - first_hit->query_seq_pos;
  double dist_t = second_hit->target_seq_pos - first_hit->target_seq_pos;
  //dist_q += 1.0;
  //dist_t += 1.0;

  *expected_cosine_dvg = (log_theta_q * dist_q) + (log_theta_t * dist_t);
  *expected_cosine_dvg = exp(*expected_cosine_dvg);



  Hit hit = {.query_seq_pos =second_hit->query_seq_pos,
             .target_seq_pos =second_hit->target_seq_pos,
             .query_bin=second_hit->query_bin,
             .target_bin=second_hit->target_bin,
             .cosine_sim=first_hit->cosine_sim * (*expected_cosine_dvg)};

  double log_p = log_pval_for_hit(&hit, args);
  if (log_p == 0) {
    *expected_cosine_dvg = 0;
  }
  return  log_p;
}

double indel_cost(double q_start, double t_start, double q_end, double t_end) {

  double delta_p = fabs((q_end - q_start) - (t_end - t_start));
  delta_p = MIN(delta_p, 1.0);
  return -(delta_p * log(0.05)); // indel cost
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
  double effective_db_chance = (double)args->hits_per_emb / (double)args->index_size;
  double effective_log_db_chance = log(effective_db_chance);
  if (N == 0)
    return 0.0; /* empty slice → p = 1 */

  double *dp =
      (N <= DP_STACK_LIM) ? args->dp_st : (double *)malloc(N * sizeof(*dp));
  float *areas =
      (N <= DP_STACK_LIM) ? args->ln_st : (float *)malloc(N * sizeof(float));


  volatile double combined_score = 1000.0; // big bug if not volatile
  volatile double remaining_area = 0;
  double best_area = 1;

  for (size_t i = 0; i < N; ++i) {
    const Hit *hi = &hits[start + i];

    double hi_hit_p = log_pval_for_hit(hi, args);
    double excluded_area = excluded_area_for_start(hi->query_pos,
                                                   hi->target_pos,
                                                   query_length,
                                                   target_length
                                                   );

    double best_hij = hi_hit_p +
                      log(query_length * target_length) +
                      effective_log_db_chance; /* path that starts at i */
    float len_i = 1;

    /* ---------- inner scan, backwards, branch-light ------------- */
    for (ssize_t j = (ssize_t)i - 1; j >= 0; --j) {
      const Hit *hj = &hits[start + j];

      /* cheap rejection first */
      if (hj->target_seq_pos == hi->target_seq_pos)
        continue; /* same col */
      if (hj->query_seq_pos >= hi->query_seq_pos)
        continue; /* wrong col */


      /* passed the two filters -> valid predecessor  */


      double expected_cosine = 0;
      double expected_hijp = expected_hit_logp(args, hj, hi, &expected_cosine);
      double indel_logp = indel_cost(hj->query_seq_pos,
                                     hj->target_seq_pos,
                                     hi->query_seq_pos,
                                     hi->target_seq_pos);
      // Now we make the excluded_area adjustment
      double cand_area = excluded_area_for_start(hj->query_pos,
                                                 hj->target_pos,
                                                 query_length,
                                                 target_length);

      double cand = dp[j] + // P of current path
                    hi_hit_p - expected_hijp + // P of hit given last hit
                    indel_logp + // P of hit given potential indels
                    log(cand_area + 1) + // P of hit given area to find hit
                    effective_log_db_chance;// - expected_hijp; // P of this hit being found



      //cand += log((cand_area + 1) * 0.5);

      if (cand < best_hij) { /* smaller -> rarer -> better -> faster ->stronger*/
        best_hij = cand;
        excluded_area = cand_area;
      }
    }

    dp[i] = best_hij;
    areas[i] = len_i;
   // printf("Checking if best_hij better than combined score: %f %p\n", combined_score, (void*)&combined_score);
    if (best_hij < combined_score) {
      combined_score = best_hij;
      best_area = excluded_area;
    }
  }

  double hits_per_emb = args->hits_per_emb;
  double expected_hits_per_emb = hits_per_emb * target_length / args->index_size;

  double lambda = combined_score;
  double log_pval = lambda;

  if (dp != args->dp_st)
    free(dp);
  if (areas != args->ln_st)
    free(areas);
  if (nhits != NULL)
    *nhits = best_area;

  return log_pval;
}

void process_hit_range(const ProcessHitArgs *args, uint64_t starting_index,
                       uint64_t ending_index) {
  QueryTargetSimilarity qt_sim;

  double query_length =
      args->query_lengths[args->hits[starting_index].query_seq_id];
  double target_length =
      args->target_lengths[args->hits[starting_index].target_seq_id];

  //printf("%f %f\n", query_length, target_length);

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
                                                           ending_index,
                                                           query_length,
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