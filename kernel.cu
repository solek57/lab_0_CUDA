__global__ void blockKernel(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p , unsigned  int block_size)
{
	extern  __shared__ float shared[];
	float* block_a = shared;
    float* block_b = (float*)&block_a[block_size * block_size];
	const int idx = blockIdx.x * block_size + threadIdx.x,
		idy = blockIdx.y * block_size + threadIdx.y,
		maxI = (p - 1) / block_size + 1,
        block_index = threadIdx.y * block_size + threadIdx.x;
	if (!(idy < n && idx < m)) {
		return;
	}
	for (int i = 0; i < maxI; i++) {
		block_a[block_index] = (idy < n && i * block_size + threadIdx.x < p) ?
			a[idy * p + i * block_size + threadIdx.x] : 0;
		block_b[block_index] = (idx < m && i * block_size + threadIdx.y < p) ?
			b[(i * block_size + threadIdx.y) * m + idx] : 0;
		__syncthreads();
		for (int k = 0; k < block_size; k++)
			c[idy * m + idx] += block_a[threadIdx.y * block_size + k] * block_b[k * block_size + threadIdx.x];
		__syncthreads();
	}
}

#define kernel blockKernel
#include "main.h"