
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <ctime>
#include <random>
#include <string.h>

#define N 4096
#define M 4096
#define K 4096
#define A_SIZE N*K
#define B_SIZE K*M
#define C_SIZE N*M
#define BLOCK_SIZE 32

cudaError_t multWithCuda(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p, unsigned  int block_size);

void fill_by_rand(float *a, unsigned int size);

void cpu_ijk(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p);

void cpu_ikj(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p);

void cpu_kij(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p);

bool checkEquals(float* a, float* b, unsigned int n, unsigned int m);

void printMatrix(float* a, unsigned int n, unsigned int m);

int main()
{
	float* a, * b, * c, * d;
	clock_t start;
	a = (float*)calloc(A_SIZE, sizeof(float));
	b = (float*)calloc(B_SIZE, sizeof(float));
	c = (float*)calloc(C_SIZE, sizeof(float));
	d = (float*)calloc(C_SIZE, sizeof(float));

	fill_by_rand(a, A_SIZE);
	fill_by_rand(b, B_SIZE);

	start = clock();
	cpu_ijk(d, a, b, N, M, K);
	printf("\nCPU's time for ijk: %f", (clock() - start) / (double) CLOCKS_PER_SEC);

	memset(d, 0, C_SIZE * sizeof(float));
	start = clock();
	cpu_ikj(d, a, b, N, M, K);
	printf("\nCPU's time for ikj: %f", (clock() - start) / (double)CLOCKS_PER_SEC);

	memset(d, 0, C_SIZE * sizeof(float));
	start = clock();
	cpu_kij(d, a, b, N, M, K);
	printf("\nCPU's time for kij: %f", (clock() - start) / (double)CLOCKS_PER_SEC);

    cudaError_t cudaStatus = multWithCuda(c, a, b, N, M, K, BLOCK_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "multWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	printf("matrixs is equals: %d\n", checkEquals(c, d, N, M));

    return 0;
}

cudaError_t multWithCuda(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p, unsigned  int block_size)
{
	float *dev_a = 0, *dev_b = 0, *dev_c = 0;
	float gpuTime = 0.0f;
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	dim3 dimGrid((M - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1, 1);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

	cudaStatus = cudaEventCreate(&start);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaStatus = cudaEventCreate(&stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventCreate failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	cudaStatus = cudaEventRecord(start);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord failed record!");
		goto Error;
	}

	cudaEventSynchronize(start);

    cudaStatus = cudaMalloc((void**)&dev_c, C_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, A_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, B_SIZE * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, B_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	blockKernel <<<dimGrid, dimBlock,  2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float) >>>(dev_c, dev_a, dev_b, N, M, K, BLOCK_SIZE);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error after star! %d\n", cudaStatus);
		goto Error;
	}

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "blockKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, C_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaEventRecord(stop);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaEventRecord failed record!");
		goto Error;
	}

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&gpuTime, start, stop);
	printf("\nGPU's time spent executing %s: %f seconds\n", "kernel", gpuTime / 1000);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

void fill_by_rand(float* a, unsigned int size) {
	for (int i = 0; i < size; i++) a[i] = (float)rand() / RAND_MAX;
}

void cpu_ijk(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p) {
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < m; j++) {
			for (unsigned int k = 0; k < p; k++) {
				c[i * m + j] += a[i * p + k] * b[k * m + j];
			}
		}
	}
}

void cpu_ikj(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p) {
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int k = 0; k < p; k++) {
			for (unsigned int j = 0; j < m; j++) {
				c[i * m + j] += a[i * p + k] * b[k * m + j];
			}
		}
	}
}

void cpu_kij(float* c, float* a, float* b, unsigned int n, unsigned int m, unsigned int p) {
	for (unsigned int k = 0; k < p; k++) {
		for (unsigned int i = 0; i < n; i++) {
			for (unsigned int j = 0; j < m; j++) {
				c[i * m + j] += a[i * p + k] * b[k * m + j];
			}
		}
	}
}

bool checkEquals(float* a, float* b, unsigned int n, unsigned int m) {
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < m; j++) {
			if (a[i * m + j] != b[i * m + j]) {
				return false;
			}
		}
	}
	return true;
}

void printMatrix(float* a, unsigned int n, unsigned int m) {
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < m; j++) {
			printf("%.3f ", a[i * m + j]);
		}
	}
	printf("\n\r");
}