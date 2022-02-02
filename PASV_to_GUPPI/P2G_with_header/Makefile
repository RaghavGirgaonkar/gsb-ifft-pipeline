CC=gcc

CFLAGS= -lm -L/usr/local/lib 
main=pasv2guppi
guppi=guppi_header


p2g: guppi_header.o pasv2guppi.o 
	$(CC) -o $(main) $(main).o $(guppi).o $(CFLAGS)

pasv2guppi.o: guppi_header.o pasv2guppi.c 
	$(CC) -c $(CFLAGS) $(main).c

guppi_header.o: guppi_header.h guppi_header.c 
	$(CC) -c $(CFLAGS) $(guppi).c 

clean:
	rm *.o