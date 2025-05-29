//
// Created by Daniel Olson on 5/29/25.
//

#ifndef PROCESS_HITS_IO_H
#define PROCESS_HITS_IO_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include "types.h"

void err_crash(const char *fmt, ...);
void output_similarity(const ProcessHitArgs *args, QueryTargetSimilarity similarity);
uint64_t get_seq_list_from_pipe(char** name_list_ptr, uint64_t** start_list_ptr, uint64_t**seq_lengths_ptr);
uint64_t get_hits_from_pipe(Hit **hit_list_ptr, int hits_per_query);
ProcessHitArgs read_arguments(int argc, const char** argv);
void           read_name_lists(ProcessHitArgs *args);

#endif //PROCESS_HITS_IO_H
