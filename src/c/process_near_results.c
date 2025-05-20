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
#include "fluxsort.h"

/// Buffer size for file I/O (32 MiB)
#define OUTPUT_BUF_SIZE   (1 << 25)
/// Threshold for reporting total scores
#define SCORE_THRESHOLD   8.0f
/// Mask to extract high 32 bits (sequence ID
#define SEQ_LABEL_MASK    0xFFFFFFFF00000000ULL

typedef struct {
    uint64_t query_seq_id;
    uint64_t target_label;
    float    score;
    uint8_t  target_pos_bin;
    uint8_t  query_pos_bin;
} IndexScore;

static int cmp_idx(const void *a, const void *b) {
    uint64_t ai = ((const IndexScore*)a)->target_label;
    uint64_t bi = ((const IndexScore*)b)->target_label;
    return (ai < bi ? -1 : (ai > bi ? 1 : 0));
}

static inline float gumbel_r_cdf(float loc, float inverse_scale, float x) {
    float z = (x - 0.32160510842266415) * 34.0281875276;
    return expf(-expf(-z));
}

static inline IndexScore* get_index_scores_from_pipe(uint64_t *num_hits_ptr) {
    /* read how many hits we have */
    if (fread(num_hits_ptr, sizeof(*num_hits_ptr), 1, stdin) != 1) {
        fprintf(stderr, "Failed to read num_hits\n");
        exit(1);
    }
    uint64_t num_hits = *num_hits_ptr;

    /* allocate */
    IndexScore *index_scores = malloc(num_hits * sizeof(IndexScore));
    uint64_t    *buf          = malloc(num_hits * sizeof(uint64_t));
    if (!index_scores || !buf) {
        perror("malloc");
        free(index_scores);
        free(buf);
        exit(1);
    }

    size_t count;
    /* read query‐labels (packed: high 32 bits = seq_id) */
    count = fread(buf, sizeof(uint64_t), num_hits, stdin);
    if (count != num_hits) {
        fprintf(stderr, "Expected %llu query labels, got %zu\n",
                (unsigned long long)num_hits, count);
        free(index_scores); free(buf);
        exit(1);
    }
    for (size_t i = 0; i < num_hits; ++i) {
        index_scores[i].query_seq_id  = (buf[i] & SEQ_LABEL_MASK) >> 32;
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
        index_scores[i].target_label = buf[i];
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
            uint8_t bin = index_scores[i].query_pos_bin;
            float loc   = gumbel_values[bin][0];
            float inv_s = gumbel_values[bin][1];
            float pval  = 1.0f - gumbel_r_cdf(loc, inv_s, raw);
            index_scores[i].score = -log2f(1e-20f + pval);
        }
        /* when query ID changes, sort the last block */
        if (index_scores[i].query_seq_id != last_query) {
            fluxsort_size(&index_scores[query_start],
                          i - query_start,
                          sizeof(IndexScore),
                          cmp_idx);
            query_start = i;
            last_query  = index_scores[i].query_seq_id;
        }
    }
    /* final block */
    fluxsort_size(&index_scores[query_start],
                  num_hits - query_start,
                  sizeof(IndexScore),
                  cmp_idx);

    free(buf);
    return index_scores;
}

static inline void output_index_scores_to_file(const char *file_path,
                                               IndexScore  *index_scores,
                                               uint64_t     num_hits) {
    FILE *out = fopen(file_path, "w");
    if (!out) { perror("fopen"); exit(1); }
    static char buffer[OUTPUT_BUF_SIZE];
    setvbuf(out, buffer, _IOFBF, sizeof(buffer));

    fprintf(out, "Query\tTarget\tScore\n");

    uint64_t last_query  = index_scores[0].query_seq_id;
    uint64_t last_target = (index_scores[0].target_label & SEQ_LABEL_MASK) >> 32;
    float    total_score = 0.0f;

    for (uint64_t i = 0; i < num_hits; ++i) {
        uint64_t q = index_scores[i].query_seq_id;
        uint64_t t = (index_scores[i].target_label & SEQ_LABEL_MASK) >> 32;

        if (q != last_query || t != last_target) {
            if (total_score > SCORE_THRESHOLD) {
                fprintf(out, "%llu\t%llu\t%f\n",
                        (unsigned long long)last_query,
                        (unsigned long long)last_target,
                        total_score);
            }
            last_query  = q;
            last_target = t;
            total_score = 0.0f;
        }
        total_score += index_scores[i].score;
    }

    /* final flush */
    if (total_score > SCORE_THRESHOLD) {
        fprintf(out, "%llu\t%llu\t%f\n",
                (unsigned long long)last_query,
                (unsigned long long)last_target,
                total_score);
    }
    fclose(out);
}

int main(void) {
    uint64_t num_hits;
    IndexScore *scores = get_index_scores_from_pipe(&num_hits);
    output_index_scores_to_file("output.txt", scores, num_hits);
    free(scores);
    return 0;
}
