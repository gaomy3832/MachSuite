viterbi: viterbi.c viterbi.h ../../common/harness.c
	$(CC) $(CFLAGS) -o viterbi viterbi.c ../../common/harness.c

run: viterbi input.data check.data
	./viterbi input.data check.data

hls: viterbi.cpp viterbi.h
	vivado_hls hls.tcl

clean:
	rm -f viterbi
