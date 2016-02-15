EXECUTABLE := bench_mp_blas
OBJS := bench.o

CFLAGS := -std=c99 -fopenmp -DUSE_OPENMP -DUSE_OPENBLAS

INCLUDES := -I/opt/openblas/include
LIBLOCS := -L/opt/openblas/lib
LDFLAGS := -lm -lopenblas

CC := gcc $(INCLUDES)

%.o: %.c %.h
	$(CC) $(CFLAGS) $(DEFS) $(INCLUDES) -c $< -o $@

$(EXECUTABLE): $(OBJS)
	$(CC) $(CFLAGS) $(DEFS) $(INCLUDES) $(OBJS) -o $@ $(LIBLOCS) $(LDFLAGS)

clean:
	-rm -f *.o $(EXECUTABLE)
