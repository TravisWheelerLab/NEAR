
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#define OUTPUT_BUF_SIZE     (1 << 25)
#define TID_TO_SEQID(x)     ((x >> 32))
#define DP_STACK_LIM        1048576
#define TID_TO_BIN(x)       (x & 0x7F)
#define TID_TO_POS(x)       ((x & 0xFFFFFFFF) >> 7)

#define LOG_LAM_SMALL       (-10.0)
#define LOG_LAM_LARGE       50.0
#define LOG_HALF            (-0.6931471805599453)

typedef struct {
    uint32_t query_seq_id;
    uint32_t target_seq_id;
    uint32_t query_pos; // includes bin
    uint32_t target_pos; // includes bin
    double    logpval;
} Hit;

typedef struct {
    FILE        *out;

    double       filter_1_logpval_threshold;
    double       filter_2_logpval_threshold;

    uint64_t    num_hits;
    Hit         *hits;

    uint64_t    *query_name_starts;
    char        *query_names;

    uint64_t    *target_name_starts;
    char        *target_names;

    uint64_t       *query_lengths;
    uint64_t       *target_lengths;

    double        index_size;
    double        hits_per_emb;
    uint32_t     sparsity;
    double        num_targets;

} ProcessHitArgs;

static int cmp_hit(const void *a, const void *b)
{
    const Hit *pa = a;
    const Hit *pb = b;

    if (pa->query_seq_id  < pb->query_seq_id)  return -1;
    if (pa->query_seq_id  > pb->query_seq_id)  return  1;

    if (pa->target_seq_id < pb->target_seq_id) return -1;
    if (pa->target_seq_id > pb->target_seq_id) return  1;

    if (pa->target_pos    < pb->target_pos)    return -1;
    if (pa->target_pos    > pb->target_pos)    return  1;

    return 0;
}

static inline uint64_t get_seq_list_from_pipe(char** name_list_ptr, uint64_t** start_list_ptr, uint64_t**seq_lengths_ptr) {
    uint64_t num_names;
    uint64_t string_size;

    if (fread(&num_names, sizeof(num_names), 1, stdin) != 1) {
        fprintf(stderr, "Failed to read num_names\n");
        exit(1);
    }
    if (fread(&string_size, sizeof(string_size), 1, stdin) != 1) {
        fprintf(stderr, "Failed to read size of string\n");
        exit(1);
    }

    uint64_t* start_list = malloc(sizeof(uint64_t) * (num_names + 1));
    char *name_list = malloc(sizeof(char) * (string_size + 1));
    if (!start_list || !name_list) {
        perror("malloc");
        free(start_list);
        free(name_list);
        exit(1);
    }
    *name_list_ptr = name_list;
    *start_list_ptr = start_list;

    uint64_t count = fread(name_list, sizeof(char), string_size, stdin);
    if (count != string_size) {
        fprintf(stderr, "Expected %llu characters while reading name list, got %llu\n",
                (unsigned long long)string_size, count);
        free(start_list);
        free(name_list);
        exit(1);
    }
    name_list[count] = '\0';

    start_list[0] = 0;
    count = 1;
    for (uint64_t i = 0; i < string_size; ++i) {
        if (name_list[i] == '\0') {
            start_list[count] = i + 1;
            ++count;
        }
    }

    start_list[count] = string_size + 1;

    if (seq_lengths_ptr != NULL) {
        uint64_t* seq_lengths = malloc(sizeof(uint64_t) * num_names);
        if (!seq_lengths) {
            perror("malloc");
            exit(1);
        }
        *seq_lengths_ptr = seq_lengths;

        count = fread(seq_lengths, sizeof(uint64_t), num_names, stdin);
        if (count != num_names) {
            fprintf(stderr, "Expected %llu lengths while reading sequence lengths, got %llu\n",
                    (unsigned long long)num_names, count);
            exit(1);
        }
    }

    return num_names;
}

static inline uint64_t get_hits_from_pipe(Hit **hit_list_ptr, int hits_per_query) {
    const double score_penalty = log2f(0.5);
// Read the number of queries and calculate the number of hits
    uint64_t num_queries;
    if (fread(&num_queries, sizeof(num_queries), 1, stdin) != 1) {
        return 0;
    }

    uint64_t num_hits = hits_per_query * num_queries;

/* allocate */
    Hit *hits = malloc(num_hits * sizeof(Hit));
    uint64_t    *buf          = malloc(num_hits * sizeof(uint64_t));
    if (!hits || !buf) {
        perror("malloc");
        free(hits);
        free(buf);
        exit(1);
    }

    *hit_list_ptr = hits;

    size_t count;
/* read query‐labels (packed: high 32 bits = seq_id) */
    count = fread(buf, sizeof(uint64_t), num_queries, stdin);
    if (count != num_queries) {
        fprintf(stderr, "Expected %llu query labels, got %zu\n",
                (unsigned long long)num_queries, count);
        free(hits); free(buf);
        exit(1);
    }
    for (size_t i = 0; i < num_queries; ++i) {
        uint32_t query_seq_id = TID_TO_SEQID(buf[i]);
        uint32_t query_pos = (uint32_t)(buf[i]);
        for (size_t j = (i*hits_per_query); j < (i*hits_per_query) + hits_per_query; ++j) {
            hits[j].query_seq_id  = query_seq_id;
            hits[j].query_pos = query_pos;
        }

    }

/* read target labels */
    count = fread(buf, sizeof(uint64_t), num_hits, stdin);
    if (count != num_hits) {
        fprintf(stderr, "Expected %llu target labels, got %zu\n",
                (unsigned long long)num_hits, count);
        free(hits); free(buf);
        exit(1);
    }
    for (size_t i = 0; i < num_hits; ++i) {
        uint32_t target_seq_id = TID_TO_SEQID(buf[i]);
        uint32_t target_pos = (uint32_t)(buf[i]);
        hits[i].target_seq_id = target_seq_id;
        hits[i].target_pos = target_pos;
    }

/* read raw double scores into same buffer */
    double *fbuf = (double*)buf;
    count = fread(fbuf, sizeof(double), num_hits, stdin);
    if (count != num_hits) {
        fprintf(stderr, "Expected %llu scores, got %zu\n",
                (unsigned long long)num_hits, count);
        free(hits); free(buf);
        exit(1);
    }
/* convert to log2 E‐values and sort per query block */
    uint64_t last_query = hits[0].query_seq_id;
    size_t   query_start = 0;
    for (size_t i = 0; i < num_hits; ++i) {
        hits[i].logpval = fbuf[i];
/* when query ID changes, sort the last block */
        if (hits[i].query_seq_id != last_query) {
            qsort(&hits[query_start],
                  i - query_start,
                  sizeof(Hit),
                  cmp_hit);
            query_start = i;
            last_query  = hits[i].query_seq_id;
        }
    }
/* Sort the last block */
    qsort(&hits[query_start],
          num_hits - query_start,
          sizeof(Hit),
          cmp_hit);

    free(buf);
    return num_hits;
}

uint64_t seqlist_size (uint64_t *seq_lengths, uint64_t num_lengths) {
    uint64_t total_embeddings = 0;
    for (uint64_t i = 0; i < num_lengths; ++i) {
        total_embeddings += seq_lengths[i];
    }
    return total_embeddings;
}

static inline double log_binom(double n, double k) {
    if (k > n - k) {
        k = n - k;
    }
    return lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0);
}

static inline double log1mexp(double x)
{
    return (x > LOG_HALF) ? log(-expm1(x))
                          : log1p(-exp(x));
}

static inline double log_rook(int a, int b, int k)
{
    return  lgamma(a + 1.0) - lgamma(a - k + 1.0) +
            lgamma(b + 1.0) - lgamma(b - k + 1.0) -
            lgamma(k + 1.0);
}

static inline double log_poisson_tail(double log_lambda)
{
    if (log_lambda <= LOG_LAM_SMALL)                 /* tiny lambda */
        return log_lambda;                           /* log p ~ log lambda        */

    if (log_lambda >= LOG_LAM_LARGE)                 /* huge lambda */
        return 0.0;                                 /* p ~ 1, log p ~ 0     */

    double lambda = exp(log_lambda);                 /* safe: lambda < e^50 */
    return log1mexp(-lambda);
}

// Filter 1 treats hits as independent
double log_pval_from_independent_hits(const Hit *hits,
                                      uint64_t   start,
                                      uint64_t   end,
                                      int        n_rows,   /* a  */
                                      int        n_cols)   /* b  */
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


static inline double log_odds_transition(uint32_t q_i, uint32_t t_i,
                                         uint32_t q_j, uint32_t t_j,
                                         double   logp_i, double   logp_j)
{

}

/* --------------------------------------------------------------------------
 *  Filter-2 : coherent-path p-value
 * --------------------------------------------------------------------------*/
static inline double
logpval_from_coherent_hits(Hit      *restrict hits,
                           uint64_t  start,
                           uint64_t  end,
                           uint64_t  n_rows,
                           uint64_t  n_cols)
{
    const size_t N = (size_t)(end - start);
    if (N == 0) return 0.0;                    /* empty slice → p = 1 */

    /* ---- scratch, stack-backed when N ≤ 1024 ------------------------ */
    double dp_st [DP_STACK_LIM];
    int    ln_st [DP_STACK_LIM];
    double *dp   = (N <= DP_STACK_LIM) ? dp_st : (double *)malloc(N*sizeof(*dp));
    int    *plen = (N <= DP_STACK_LIM) ? ln_st : (int    *)malloc(N*sizeof(*plen));

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
    double log_lambda = best_score
                        + log_rook((int)n_rows, (int)n_cols, best_len);
    double log_pval   = log_poisson_tail(log_lambda);

    if (dp != dp_st)   free(dp);
    if (plen != ln_st) free(plen);
    return log_pval;                       /* already log(p-value) */
}

// Filter 2 considers a path through the hits that is coherent
static inline double losssgpval_from_coherent_hits(Hit *hits,
                                            uint64_t start,
                                            uint64_t end,
                                            uint64_t query_length,
                                            uint64_t target_length) {
    /* the hit structure
     * typedef struct {
    uint32_t query_seq_id;
    uint32_t target_seq_id;
    uint32_t query_pos; // includes bin
    uint32_t target_pos; // includes bin
    double    logpval;
} Hit;
     * */
    // The goal of this function is to calculate a maximal path
    // through hits.
    // The path is not allowed to re-use query_pos/target_pos
    // The path is not allowed to traverse to decreasing query_pos/target_pos
    // All transitions will have an associated log odds value that gets added to query_pos/targetpos
    // The transition score may be positive or negative
    // Transitions can be calculated via the function
    // double log_odds_transition(query_pos_i,
                                // target_pos_i,
                                // query_pos_j,
                                // target_pos_j,
                                // hiti_logpval
                                // hitj_logpval)

    // By default, hits[start:end] is sorted in ascending order of target_pos
    // although there will never be two hits with the same target pos AND query pos
    // there may be hits with the same target pos or the same query pos

    // Write the code here.

    return 0;
}

void output_qt_pair(uint32_t query_id,
                    uint32_t target_id,
                    double filter1_pval,
                    double filter2_pval,
                    ProcessHitArgs args) {

}

static inline void process_hit_range(uint64_t starting_index,
                                    uint64_t ending_index,
                                    ProcessHitArgs args) {
    uint64_t query_length = args.query_lengths[args.hits[starting_index].query_seq_id];
    uint64_t target_length = args.target_lengths[args.hits[starting_index].target_seq_id];
    // Calculate first filter pval
    double qt_filter1_logpval = log_pval_from_independent_hits(args.hits,
                                                          starting_index,
                                                          ending_index,
                                                          query_length,
                                                          target_length);
    if (qt_filter1_logpval < args.filter_1_logpval_threshold) {

        // If it passes first filter, calculate second filter pval
        double qt_filter2_logpval = pval_from_coherent_hits(args.hits,
                                                           starting_index,
                                                           ending_index,
                                                           query_length,
                                                           target_length);
        if (qt_filter2_logpval < args.filter_2_logpval_threshold) {
            // Output the query target pair it passes the second filter
            output_qt_pair(args.hits[starting_index].query_seq_id,
                           args.hits[starting_index].target_seq_id,
                           qt_filter1_logpval,
                           qt_filter2_logpval, args);
        }
    }
}

static inline void output_index_scores_to_file(ProcessHitArgs args) {

    Hit *hits = args.hits;

    uint32_t last_query  = hits[0].query_seq_id;
    uint32_t last_target = hits[0].target_seq_id;

    uint64_t starting_index = 0;

    for (uint64_t i = 1; i < args.num_hits; ++i) {
        uint32_t q  = hits[i].query_seq_id;
        uint32_t t  = hits[i].target_seq_id;

        if (q != last_query || t != last_target) {
            process_hit_range(starting_index, i, args);

            starting_index = i;
            last_query = q;
            last_target = t;
        }
    }

    // final flush
    if (last_query != (uint32_t)-1) {
        process_hit_range(starting_index, args.num_hits, args);
    }
}



int main(int argc, const char** argv) {
    printf("Opening file\n");
    FILE *out = fopen(argv[1], "w");

    if (!out) { perror("fopen"); exit(1); }
    static char buffer[OUTPUT_BUF_SIZE];
    setvbuf(out, buffer, _IOFBF, sizeof(buffer));

    fprintf(out, "Query\tTarget\tp-val\te-val\tHits\n");

    int hits_per_emb = atoi(argv[2]);
    double score_threshold = (double)atof(argv[3]);
    int sparsity = atoi(argv[4]);

    uint64_t num_query_names;
    uint64_t num_target_names;
    Hit *hits;

    char *query_names;
    char *target_names;
    uint64_t *query_name_starts;
    uint64_t *target_name_starts;

    uint64_t *query_lengths;
    uint64_t *target_lengths;

    printf("Reading query names...\n");
    num_query_names = get_seq_list_from_pipe(&query_names, &query_name_starts, &query_lengths);
    printf("Reading target names...\n");
    num_target_names = get_seq_list_from_pipe(&target_names, &target_name_starts, &target_lengths);

    double index_size = seqlist_size(target_lengths, num_target_names);
    ProcessHitArgs args;
    args.out = out;
    args.query_name_starts = query_name_starts;
    args.query_names = query_names;
    args.target_name_starts = target_name_starts;
    args.target_names = target_names;
    args.filter_1_logpval_threshold = score_threshold;
    args.filter_2_logpval_threshold = score_threshold;
    args.query_lengths = query_lengths;
    args.target_lengths = target_lengths;
    args.index_size = index_size;
    args.hits_per_emb = hits_per_emb;
    args.sparsity = sparsity;
    args.num_targets = num_target_names;

    while (1) {
        printf("Reading hits...\n");
        uint64_t num_hits = get_hits_from_pipe(&hits, hits_per_emb);
        if (num_hits == 0) {
            break;
        }
        args.num_hits = num_hits;
        args.hits = hits;
        printf("Outting hits...\n");
        output_index_scores_to_file(args);
        free(hits);
    }
    printf("Done.\n");
    free(query_names);
    free(target_names);
    free(query_name_starts);
    free(target_name_starts);
    fclose(out);
    return 0;
}
