#include "io.h"
#include "process_hits.h"
#include "types.h"
#include "util.h"

int main(int argc, const char **argv) {
  DPRINTF("Reading args\n");
  ProcessHitArgs args = read_arguments(argc, argv);

  DPRINTF("Reading name list\n");
  read_name_lists(&args);

  static char buffer[OUTPUT_BUF_SIZE];
  setvbuf(args.out, buffer, _IOFBF, sizeof(buffer));

  if (args.n_threads == 1) {
    while (1) {
      Hit *hits;
      DPRINTF("Getting hits\n");
      uint64_t num_hits = get_hits_from_pipe(&hits, args.hits_per_emb);
      if (num_hits == 0) {
        DPRINTF("no hits\n");
        break;
      }
      args.hits = hits;
      args.num_hits = num_hits;
      DPRINTF("Processing hits\n");
      process_hits(args);
      fflush(args.out);
      free(hits);
    }
  } else {
  }

  fclose(args.out);
  return 0;
}