#include "types.h"
#include "util.h"
#include "io.h"
#include "process_hits.h"

int main(int            argc,
         const char**   argv)
{
    ftz_enable(); // Disable denormals
    ProcessHitArgs args = read_arguments(argc, argv);
    read_name_lists(&args);

    static char buffer[OUTPUT_BUF_SIZE];
    setvbuf(args.out, buffer, _IOFBF, sizeof(buffer));

    if (args.n_threads == 1) {
        while (1) {
            Hit *hits;
            uint64_t num_hits = get_hits_from_pipe(&hits,
                                                   args.hits_per_emb);
            if (num_hits == 0) {
                break;
            }
            args.hits = hits;
            args.num_hits = num_hits;
            process_hits(args);
            free(hits);
        }
    } else {

    }

    fclose(args.out);
    return 0;
}