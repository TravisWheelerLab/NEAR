/**
 *
 * Written by Daniel Olson on May 16th
 *
 * Program to read index scores from stdin,
 *  interpret scores as coming from a Gumbel distribution,
 *      combine query->target scores,
 *      output results to a file.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "gumbel_parameters.h"

#include <float.h>

/**
 * Compute log(1 - exp(x)) in a numerically stable way for x <= 0.
 *
 * For x == 0, returns -INFINITY (log(0)).
 * For x <= ln(0.5), uses log1p(-exp(x)).
 * For ln(0.5) < x < 0, uses log(-expm1(x)).
 * For x > 0, returns NAN.
 */
float log1mexp(float x) {
    const float LN_HALF = -0.6931471805599453;  /* ln(0.5) */

    if (x == 0.0) {
        /* log(1 - 1) = log(0) = -inf */
        return -INFINITY;
    }
    else if (x <= LN_HALF) {
        /* exp(x) is small enough that 1 - exp(x) is not too close to 1 */
        return log1p(-exp(x));
    }
    else if (x < 0.0) {
        /* expm1(x) = exp(x) - 1 is close to -small, so -expm1(x) is small positive */
        return log(-expm1(x));
    }
    else {
        /* undefined / out-of-domain (x > 0) */
        return NAN;
    }
}

/// Buffer size for file I/O (32 MiB)
#define OUTPUT_BUF_SIZE   (1 << 25)
/// Mask to extract high 32 bits (sequence ID
#define SEQ_LABEL_MASK    0xFFFFFFFF00000000ULL
/// Mask for stat bin
#define STAT_BIN_MASK     0x7F

#define FLAT_HIT_SCORE    4.0

typedef struct {
    uint32_t query_seq_id;
    uint32_t target_seq_id;
    uint32_t query_pos;
    uint32_t target_pos;
    float    score;
} IndexScore;

static int cmp_idx(const void *a, const void *b)
{
    const IndexScore *pa = a;
    const IndexScore *pb = b;

    if (pa->query_seq_id  < pb->query_seq_id)  return -1;
    if (pa->query_seq_id  > pb->query_seq_id)  return  1;

    if (pa->target_seq_id < pb->target_seq_id) return -1;
    if (pa->target_seq_id > pb->target_seq_id) return  1;

    if (pa->target_pos    < pb->target_pos)    return -1;
    if (pa->target_pos    > pb->target_pos)    return  1;

    return 0;
}

static inline float gumbel_r_cdf(float loc, float inverse_scale, float x) {
    float z = (x - loc) * inverse_scale;
    return expf(-expf(-z));
}

static inline uint64_t get_name_list_from_pipe(char** name_list_ptr, uint64_t** start_list_ptr, uint64_t**seq_lengths_ptr) {
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

// Note, this uses the same memory
static inline float convert_lengths_to_scores(uint64_t *seq_lengths, uint64_t num_lengths) {
    float *scores = (float*)seq_lengths;
    float total_embeddings = 0;
    for (uint64_t i = 0; i < num_lengths; ++i) {
        total_embeddings += seq_lengths[i];
        scores[i] = seq_lengths[i];
    }
    return total_embeddings;
}

static inline uint64_t get_index_scores_from_pipe(IndexScore **index_score_ptr, int hits_per_query) {
    const float score_penalty = log2f(0.5);
// Read the number of queries and calculate the number of hits
    uint64_t num_queries;
    if (fread(&num_queries, sizeof(num_queries), 1, stdin) != 1) {
        return 0;
    }

    uint64_t num_hits = hits_per_query * num_queries;

/* allocate */
    IndexScore *index_scores = malloc(num_hits * sizeof(IndexScore));
    uint64_t    *buf          = malloc(num_hits * sizeof(uint64_t));
    if (!index_scores || !buf) {
        perror("malloc");
        free(index_scores);
        free(buf);
        exit(1);
    }

    *index_score_ptr = index_scores;

    size_t count;
/* read query‐labels (packed: high 32 bits = seq_id) */
    count = fread(buf, sizeof(uint64_t), num_queries, stdin);
    if (count != num_queries) {
        fprintf(stderr, "Expected %llu query labels, got %zu\n",
                (unsigned long long)num_queries, count);
        free(index_scores); free(buf);
        exit(1);
    }
    for (size_t i = 0; i < num_queries; ++i) {
        uint32_t query_seq_id = (buf[i] & SEQ_LABEL_MASK) >> 32;
        uint32_t query_pos = (uint32_t)(buf[i]);
        for (size_t j = (i*hits_per_query); j < (i*hits_per_query) + hits_per_query; ++j) {
            index_scores[j].query_seq_id  = query_seq_id;
            index_scores[j].query_pos = query_pos;
        }

    }

/* read target labels */
    count = fread(buf, sizeof(uint64_t), num_hits, stdin);
    if (count != num_hits) {
        fprintf(stderr, "Expected %llu target labels, got %zu\n",
                (unsigned long long)num_hits, count);
        free(index_scores); free(buf);
        exit(1);
    }
    for (size_t i = 0; i < num_hits; ++i) {
        uint32_t target_seq_id = (buf[i] & SEQ_LABEL_MASK) >> 32;
        uint32_t target_pos = (uint32_t)(buf[i]);
        index_scores[i].target_seq_id = target_seq_id;
        index_scores[i].target_pos = target_pos;
    }

/* read raw float scores into same buffer */
    float *fbuf = (float*)buf;
    count = fread(fbuf, sizeof(float), num_hits, stdin);
    if (count != num_hits) {
        fprintf(stderr, "Expected %llu scores, got %zu\n",
                (unsigned long long)num_hits, count);
        free(index_scores); free(buf);
        exit(1);
    }
/* convert to log2 E‐values and sort per query block */
    uint64_t last_query = index_scores[0].query_seq_id;
    size_t   query_start = 0;
    for (size_t i = 0; i < num_hits; ++i) {
        float raw = fbuf[i];
        if (raw < 0 || isnan(raw)) {
            index_scores[i].score = 0;
        } else {
            uint8_t bin_q = index_scores[i].query_pos & STAT_BIN_MASK;
            uint8_t bin_t = index_scores[i].target_pos & STAT_BIN_MASK;
            float loc   = gumbel_r_params[bin_q][bin_t][0];
            float inv_s = gumbel_r_params[bin_q][bin_t][1];
            float pval  = gumbel_r_cdf(loc, inv_s, raw);
            float score = score_penalty - log2f(1e-20f + pval) ;
            if (score < 0)
                score = 0;
            index_scores[i].score = raw; //logf(1e-60 + pval);
        }
/* when query ID changes, sort the last block */
        if (index_scores[i].query_seq_id != last_query) {
            qsort(&index_scores[query_start],
              i - query_start,
              sizeof(IndexScore),
              cmp_idx);
            query_start = i;
            last_query  = index_scores[i].query_seq_id;
        }
    }
/* Sort the last block */
    qsort(&index_scores[query_start],
                  num_hits - query_start,
                  sizeof(IndexScore),
                  cmp_idx);

    free(buf);
    return num_hits;
}

static inline float score_to_logp(float score, float q_length, float t_length) {
        return log1mexp(score * q_length * t_length);
}


static inline void output_index_scores_to_file(FILE        *out,
                                               uint64_t     num_hits,
                                               IndexScore  *index_scores,
                                               uint64_t    *query_name_starts,
                                               char        *query_names,
                                               uint64_t    *target_name_starts,
                                               char        *target_names,
                                               float        score_threshold,
                                               float       *query_lengths,
                                               float       *target_lengths,
                                               float        index_size,
                                               float        hits_per_emb,
                                               uint32_t     sparsity,
                                               float        num_targets       ) {

    uint32_t last_query  = -1;
    uint32_t last_target = -1;
    uint32_t query_length = 0;
    uint32_t target_length = 0;

    uint32_t last_target_pos = -1;
    float    effective_index_size = index_size;
    float    hit_score   = -100.0f;
    float    total_score = -100.0f;
    float    largest_tp_score = -100.0f;

    uint64_t nhits = 0;

    for (uint64_t i = 0; i < num_hits; ++i) {
        uint32_t q = index_scores[i].query_seq_id;
        uint32_t t = index_scores[i].target_seq_id;
        uint32_t tp = index_scores[i].target_pos;

        float score = index_scores[i].score;

        if (q != last_query || t != last_target) {
            if (total_score > score_threshold) {
                fprintf(out, "%s\t%s\t%f\t%f\t%llu\n",
                        &query_names[query_name_starts[last_query]],
                        &target_names[target_name_starts[last_target]],
                        expf(total_score), expf(total_score)*num_targets, nhits);
            }
            query_length = query_lengths[q];
            target_length = target_lengths[t];

            last_target_pos = tp;
            last_query  = q;
            last_target = t;

            largest_tp_score = score;

            total_score = score_to_logp(score, query_length, target_length);
            nhits = 1;
        }
        else {
            score = score_to_logp(score, query_length, target_length);
            if (last_target_pos == tp) {
                if (index_scores[i].score > largest_tp_score) {
                    total_score -= score_to_logp(largest_tp_score, query_length - ((nhits - 1) * sparsity), target_length - (nhits - 1));
                    total_score += score_to_logp(score, query_length - ((nhits - 1) * sparsity), target_length - (nhits - 1));
                    largest_tp_score = score;
                }
            }

            else {
                last_target_pos = tp;
                largest_tp_score = score;
                total_score += score_to_logp(score, query_length - (nhits * sparsity), target_length - nhits);
                nhits += 1;
            }
        }
    }

    /* final flush */
    if (total_score > score_threshold) {
        fprintf(out, "%s\t%s\t%f\t%f\t%llu\n",
                        &query_names[query_name_starts[last_query]],
                        &target_names[target_name_starts[last_target]],
                        expf(total_score), expf(total_score)*num_targets, nhits);
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
    float score_threshold = (float)atof(argv[3]);
    int sparsity = atoi(argv[4]);

    uint64_t num_query_names;
    uint64_t num_target_names;
    IndexScore *scores;

    char *query_names;
    char *target_names;
    uint64_t *query_name_starts;
    uint64_t *target_name_starts;

    uint64_t *query_lengths;
    uint64_t *target_lengths;

    printf("Reading query names...\n");
    num_query_names = get_name_list_from_pipe(&query_names, &query_name_starts, &query_lengths);
    printf("Reading target names...\n");
    num_target_names = get_name_list_from_pipe(&target_names, &target_name_starts, &target_lengths);

    convert_lengths_to_scores(query_lengths, num_query_names);
    float index_size = convert_lengths_to_scores(target_lengths, num_target_names);

    while (1) {
        printf("Reading hits...\n");
        uint64_t num_hits = get_index_scores_from_pipe(&scores, hits_per_emb);
        if (num_hits == 0) {
            break;
        }
        printf("Outting hits...\n");
        output_index_scores_to_file(out,
                                    num_hits,
                                    scores,
                                    query_name_starts,
                                    query_names,
                                    target_name_starts,
                                    target_names,
                                    score_threshold,
                                    (float *)query_lengths,
                                    (float *)target_lengths,
                                    index_size,
                                    hits_per_emb,
                                    sparsity,
                                    num_target_names);
        free(scores);
    }
    printf("Done.\n");
    free(query_names);
    free(target_names);
    free(query_name_starts);
    free(target_name_starts);
    fclose(out);
    return 0;
}
