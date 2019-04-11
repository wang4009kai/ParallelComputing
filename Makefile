NVCC = nvcc
CUDAPATH = /usr/local/cuda

assignment:
	$(NVCC) assignment2.cu -o assignment2.o

run:
	$(EXEC) ./assignment2.o

clean:
	rm assignment2.o
