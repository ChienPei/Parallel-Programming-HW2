CC = gcc
CXX = g++
LDLIBS = -lpng
# CFLAGS = -lm -O3
# CFLAGS = -lm -O3 -pthread -mavx512f -march=native -g
CFLAGS = -g -O3 -pthread -mavx512f
# hw2a: CFLAGS += -pthread
hw2a: CFLAGS += -pthread -mavx512f -march=native
hw2b: CC = mpicc
hw2b: CXX = mpicxx
# hw2b: CFLAGS += -fopenmp
hw2b: CFLAGS += -fopenmp -mavx512f -mavx512dq -march=native
# -mavx512dq -march=native
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
