MAIN = main
OBJS = main.o csr.o ell.o

CC = g++
CFLAGS = -fpermissive -O2 

all: $(MAIN) 

$(MAIN): $(OBJS)
	$(CC) -o $@ $^

.SUFFIXES: .c .o
.c.o:
	$(CC) $(CFLAGS) -c $<

.PHONY: clean
clean:
	rm $(MAIN) *.o
