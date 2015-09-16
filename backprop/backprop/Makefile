net: net.c net.h ../../common/harness.c
	$(CC) $(CFLAGS) -o net net.c ../../common/harness.c -lm

run: net input.data check.data
	./net input.data check.data

hls: net.cpp net.h
	vivado_hls hls.tcl

clean:
	rm -f net
