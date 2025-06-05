gcc -DDEBUG=1 -fno-omit-frame-pointer -fsanitize=address -O1 -g -O2 -o proc main.c process_hits.c io.c util.c types.c -lm
