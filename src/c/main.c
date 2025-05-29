#include "types.h"
#include "util.h"
#include "io.h"
#include "process_hits.h"


int main(int            argc,
         const char**   argv)
{
    ftz_enable(); // Disable denormals
    printf("Reading arguments\n");
    ProcessHitArgs args = read_arguments(argc, argv);
    printf("Reading names\n");
    read_name_lists(&args);

    if (args.n_threads == 1) {
        while (1) {
            printf("Reading hits\n");
            Hit *hits;
            uint64_t num_hits = get_hits_from_pipe(&hits,
                                                   args.hits_per_emb);
            if (num_hits == 0) {
                break;
            }
            args.hits = hits;
            args.num_hits = num_hits;
            printf("Processing hits\n");
            process_hits(args);
            free(hits);
        }
    } else {

    }

    return 0;
}