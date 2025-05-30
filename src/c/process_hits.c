//
// Created by Daniel Olson on 5/29/25.
//

#include "process_hits.h"
#include "util.h"
#include "io.h"

// Filter 1 treats hits as independent
double log_pval_from_independent_hits(const Hit* restrict   hits,
                                      uint64_t              start,
                                      uint64_t              end,
                                      int                   n_rows,
                                      int                   n_cols)
{
    /* keep only the rarest hit per target */
    uint32_t last_tid      = hits[start].target_pos;
    double   tid_best_logp = hits[start].logpval;
    double   logp_sum      = tid_best_logp;
    int      nhits         = 1;

    for (uint64_t i = start + 1; i < end; ++i) {
        if (hits[i].target_pos == last_tid) {            /* same target */
            if (hits[i].logpval < tid_best_logp) {
                logp_sum      += hits[i].logpval - tid_best_logp;
                tid_best_logp  = hits[i].logpval;
            }
        } else {                                         /* new target  */
            last_tid        = hits[i].target_pos;
            tid_best_logp   = hits[i].logpval;
            logp_sum       += tid_best_logp;
            ++nhits;
        }
    }

    double log_lambda = logp_sum + log_rook(n_rows, n_cols, nhits);
    return log_poisson_tail(log_lambda);
}


double log_odds_transition(uint32_t q_i,
                           uint32_t t_i,
                           uint32_t q_j,
                           uint32_t t_j,
                           double   logp_i,
                           double   logp_j)
{
    return 0;
}

/* --------------------------------------------------------------------------
 *  Filter-2 : coherent-path p-value
 * --------------------------------------------------------------------------*/
double log_pval_from_coherent_hits(const ProcessHitArgs*    args,
                                    uint64_t                start,
                                    uint64_t                end,
                                    uint64_t                n_rows,
                                    uint64_t                n_cols)
{

    const Hit * restrict hits = args->hits;
    const size_t N = (size_t)(end - start);
    if (N == 0) return 0.0;                    /* empty slice → p = 1 */

    /* ---- scratch, stack-backed when N ≤ 1024 ------------------------ */

    double *dp   = (N <= DP_STACK_LIM) ? args->dp_st : (double *)malloc(N*sizeof(*dp));
    int    *plen = (N <= DP_STACK_LIM) ? args->ln_st : (int    *)malloc(N*sizeof(*plen));

    double best_score = hits[start].logpval;
    int    best_len   = 1;

    for (size_t i = 0; i < N; ++i) {
        const Hit *restrict hi  = &hits[start + i];

        double best_i = hi->logpval;  /* path that starts at i */
        int    len_i  = 1;

        /* ---------- inner scan, backwards, branch-light ------------- */
        for (ssize_t j = (ssize_t)i - 1; j >= 0; --j) {
            const Hit *restrict hj = &hits[start + j];

            /* cheap rejection first */
            if (hj->target_pos == hi->target_pos)   continue;   /* same col */
            if (hj->query_pos  >= hi->query_pos)    continue;   /* wrong col */

            /* passed the two filters -> valid predecessor */
            double trans = log_odds_transition(hj->query_pos, hj->target_pos,
                                               hi->query_pos, hi->target_pos,
                                               hj->logpval,   hi->logpval);

            double cand  = dp[j] + trans + hi->logpval;
            if (cand < best_i) {         /* smaller -> rarer -> better -> faster ->stronger*/
                best_i = cand;
                len_i  = plen[j] + 1;
            }
        }

        dp[i]   = best_i;
        plen[i] = len_i;

        if (best_i < best_score) { best_score = best_i; best_len = len_i; }
    }

    /* convert path score to Poisson tail, same as Filter-1 */
    double log_lambda = best_score + log_rook((int)n_rows, (int)n_cols, best_len);
    double log_pval   = log_poisson_tail(log_lambda);

    if (dp != args->dp_st)   free(dp);
    if (plen != args->ln_st) free(plen);

    return log_pval;                       /* already log(p-value) */
}

void process_hit_range(const ProcessHitArgs*     args,
                                     uint64_t           starting_index,
                                     uint64_t           ending_index) {
    QueryTargetSimilarity qt_sim;

    uint64_t query_length = args->query_lengths[args->hits[starting_index].query_seq_id];
    uint64_t target_length = args->target_lengths[args->hits[starting_index].target_seq_id];

    // Calculate first filter pval
    qt_sim.log_pval_filter_1 = log_pval_from_independent_hits(args->hits,
                                                                 starting_index,
                                                                 ending_index,
                                                                 query_length,
                                                                 target_length);
    if (qt_sim.log_pval_filter_1 < args->filter_1_logpval_threshold) {
        // If it passes first filter, calculate second filter pval
        qt_sim.log_pval_filter_2 = log_pval_from_coherent_hits(args,
                                                                  starting_index,
                                                                  ending_index,
                                                                  query_length,
                                                                  target_length);
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

    uint32_t last_query  = hits[0].query_seq_id;
    uint32_t last_target = hits[0].target_seq_id;

    uint64_t starting_index = 0;

    for (uint64_t i = 1; i < args.num_hits; ++i) {
        uint32_t q  = hits[i].query_seq_id;
        uint32_t t  = hits[i].target_seq_id;

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