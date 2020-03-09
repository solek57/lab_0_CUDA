NVCC = nvcc
CFLAGS = -g -G -O0
matrix_multiplication: kernel.cu main.h
	$(NVCC) $(CFLAGS) $< -o $@