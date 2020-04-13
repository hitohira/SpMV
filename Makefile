MAIN = main
OBJS = main.o csr.o ell.o

CXX = g++
CXXFLAGS = -fpermissive -O2 -std=c++11

all: $(MAIN) 

$(MAIN): $(OBJS)
	$(CXX) -o $@ $^

.SUFFIXES: .c .o
.c.o:
	$(CXX) $(CXXFLAGS) -c $<

.PHONY: clean
clean:
	rm $(MAIN) *.o
