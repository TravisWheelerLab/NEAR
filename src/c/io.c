//
// Created by Daniel Olson on 5/29/25.
//

#include "io.h"
#include "util.h"

void err_crash(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);

  fprintf(stderr, fmt, args);
  fprintf(stderr, "\n");

  va_end(args);

  exit(EXIT_FAILURE);
}

void output_similarity(const ProcessHitArgs *args,
                       QueryTargetSimilarity qt_sim) {
  const char *query_name =
      &args->query_names[args->query_name_starts[qt_sim.query_seq_id]];
  const char *target_name =
      &args->target_names[args->target_name_starts[qt_sim.target_seq_id]];

  fprintf(args->out, "%s\t%s\t%.5e\t%.5e\t%.5e\n", query_name, target_name,
          exp(qt_sim.log_pval_filter_1), exp(qt_sim.log_pval_filter_2),
          exp(qt_sim.log_pval_filter_2 + log(args->num_target_seqs)));
}

void get_doubles_from_pipe(double *values, uint64_t num_values) {
  uint64_t values_read = 0;
  if (fread(values, sizeof(double), 1, stdin) != num_values) {
    fprintf(stderr, "Failed to read doubles\n");
    exit(1);
  }
}

uint64_t get_seq_list_from_pipe(char **name_list_ptr, uint64_t **start_list_ptr,
                                uint64_t **seq_lengths_ptr) {
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

  uint64_t *start_list = malloc(sizeof(uint64_t) * (num_names + 1));
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
    fprintf(stderr,
            "Expected %llu characters while reading name list, got %llu\n",
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
    uint64_t *seq_lengths = malloc(sizeof(uint64_t) * num_names);
    if (!seq_lengths) {
      perror("malloc");
      exit(1);
    }
    *seq_lengths_ptr = seq_lengths;

    count = fread(seq_lengths, sizeof(uint64_t), num_names, stdin);
    if (count != num_names) {
      fprintf(
          stderr,
          "Expected %llu lengths while reading sequence lengths, got %llu\n",
          (unsigned long long)num_names, count);
      exit(1);
    }
  }

  return num_names;
}

uint64_t get_hits_from_pipe(Hit **hit_list_ptr, uint32_t hits_per_query) {

  // Read the number of queries and calculate the number of hits
  uint64_t num_queries;
  if (fread(&num_queries, sizeof(num_queries), 1, stdin) != 1) {
    return 0;
  }
  uint64_t num_hits = hits_per_query * num_queries;

  /* allocate */
  Hit *hits = malloc(num_hits * sizeof(Hit));
  uint64_t *buf = malloc(num_hits * sizeof(uint64_t));
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
    free(hits);
    free(buf);
    exit(1);
  }
  for (size_t i = 0; i < num_queries; ++i) {
    uint32_t query_seq_id = TID_TO_SEQID(buf[i]);
    uint32_t query_pos = (uint32_t)(buf[i]);
    for (size_t j = (i * hits_per_query);
         j < (i * hits_per_query) + hits_per_query; ++j) {
      hits[j].query_seq_id = query_seq_id;
      hits[j].query_pos = TID_TO_POS(query_pos);
      hits[j].query_bin = TID_TO_BIN(query_pos);
    }
  }

  /* read target labels */
  count = fread(buf, sizeof(uint64_t), num_hits, stdin);
  if (count != num_hits) {
    fprintf(stderr, "Expected %llu target labels, got %zu\n",
            (unsigned long long)num_hits, count);
    free(hits);
    free(buf);
    exit(1);
  }
  for (size_t i = 0; i < num_hits; ++i) {
    uint32_t target_seq_id = TID_TO_SEQID(buf[i]);
    uint32_t target_pos = (uint32_t)(buf[i]);

    hits[i].target_seq_id = target_seq_id;

    hits[i].target_pos = TID_TO_POS(target_pos);
    hits[i].target_bin = TID_TO_BIN(target_pos);
  }

  /* read raw double scores into the same buffer */
  double *fbuf = (double *)buf;
  count = fread(fbuf, sizeof(double), num_hits, stdin);
  if (count != num_hits) {
    fprintf(stderr, "Expected %llu scores, got %zu\n",
            (unsigned long long)num_hits, count);
    free(hits);
    free(buf);
    exit(1);
  }
  /* convert to log2 E‐values and sort per query block */
  uint64_t last_query = hits[0].query_seq_id;
  size_t query_start = 0;
  for (size_t i = 0; i < num_hits; ++i) {
    hits[i].cosine_sim = fbuf[i];
    /* when query ID changes, sort the last block */
    if (hits[i].query_seq_id != last_query) {
      qsort(&hits[query_start], i - query_start, sizeof(Hit), cmp_hit);
      query_start = i;
      last_query = hits[i].query_seq_id;
    }
  }
  /* Sort the last block */
  qsort(&hits[query_start], num_hits - query_start, sizeof(Hit), cmp_hit);

  free(buf);
  return num_hits;
}

ProcessHitArgs read_arguments(int argc, const char **argv) {
  if (argc != 7)
    err_crash("Recieved %i args.\n Usage: %s <output_file> <num_hits> <filter1 threshold> <filter2 threshold> <sparsity> "
              "<num_stats_bins> <flat_log_addition>\n",
              argc,
              argv[0]);
  ProcessHitArgs args;
  args.out = fopen(argv[1], "w");

  if (!args.out) {
    perror("fopen");
    exit(EXIT_FAILURE);
  }
  fprintf(args.out, "Query\tTarget\tfilter1_pvalue\tfilter2_pvalue\tevalue\n");
  args.hits_per_emb = atoi(argv[2]);

  args.filter_1_logpval_threshold = atof(argv[3]);
  args.filter_2_logpval_threshold = atof(argv[4]);
  args.sparsity = atoi(argv[5]);

  args.num_stat_bins = atoi(argv[6]);
  if (args.num_stat_bins != 128)
    err_crash("Only 128 bins supported for now\n");

  args.genpareto_locs = (double *)malloc(sizeof(double) * args.num_stat_bins * args.num_stat_bins);
  args.genpareto_scales = (double *)malloc(sizeof(double) * args.num_stat_bins * args.num_stat_bins);
  args.genpareto_shapes = (double *)malloc(sizeof(double) * args.num_stat_bins * args.num_stat_bins);
  args.expected_log_cosine_dvg = (double *)malloc(sizeof(double) * args.num_stat_bins);
  if (!args.genpareto_locs    ||
      !args.genpareto_scales  ||
      !args.genpareto_shapes   ||
      !args.expected_log_cosine_dvg)
  {
    perror("malloc");
    exit(1);
  }

  get_doubles_from_pipe(args.genpareto_locs,
                        args.num_stat_bins*args.num_stat_bins);
  get_doubles_from_pipe(args.genpareto_scales,
                        args.num_stat_bins*args.num_stat_bins);
  get_doubles_from_pipe(args.genpareto_shapes,
                        args.num_stat_bins*args.num_stat_bins);

  args.flat_log_addition = atof(argv[7]);

  args.n_threads = 1;
  args.thread_id = 0;

  args.dp_st = (double *)malloc(sizeof(double) * DP_STACK_LIM);
  args.ln_st = (int *)malloc(sizeof(int) * DP_STACK_LIM);

  if (!args.dp_st || !args.ln_st) {
    perror("malloc");
    exit(1);
  }

  return args;
}

void read_name_lists(ProcessHitArgs *args) {
  char *query_names;
  char *target_names;
  uint64_t *query_name_starts;
  uint64_t *target_name_starts;

  uint64_t *query_lengths;
  uint64_t *target_lengths;

  args->num_query_seqs =
      get_seq_list_from_pipe(&query_names, &query_name_starts, &query_lengths);
  args->num_target_seqs = get_seq_list_from_pipe(
      &target_names, &target_name_starts, &target_lengths);

  args->query_names = query_names;
  args->target_names = target_names;
  args->query_name_starts = query_name_starts;
  args->target_name_starts = target_name_starts;
  args->query_lengths = query_lengths;
  args->target_lengths = target_lengths;
  args->index_size = seqlist_size(args->target_lengths, args->num_target_seqs);

}

void print_arg(ProcessHitArgs args) {
  printf("num hits %llu \n", args.num_hits);
  printf("num query %llu \n", args.num_query_seqs);
  printf("num target %llu \n", args.num_target_seqs);
  printf("index size %llu \n", args.index_size);
  printf("hits per emb %u \n", args.hits_per_emb);
}