# Matrix multiplication

## Overview

Matrix multiplication with CUDA (not only square). Here is a block implementation for CUDA.
Conducted a research of the speed of execution between the GPU(CUDA) and several commom CPU algorithms.

## Usage

1. Create executable program

```console
    make matrix_multiplication
```

2. Run

```console
    ./matrix_multiplication
```

## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | Intel® Core™ i7-8750H CPU @ 2.20GHz (Turbo Boost  4.10 GHz) × 12 |
| RAM  | 16 GB DDR4 |
| GPU  | GeForce GTX 1060 with Max-Q Design/PCIe/SSE2 |
| OS type | 64-bit  |

## Conclution

Results of reseach. "SIZE" is mean multiplier size.
Average time in seconds over 10 measurements.
Elements of matrix have float type.

| SIZE      | CPU (ijk) | CPU (ikj) | CPU (kij) | GPU (BLOCK SIZE = 32) | GPU (BLOCK SIZE = 16) |
|-----------|-----------|-----------|-----------|-----------------------|-----------------------|
| 1024x1024 |   7.2386  |   3.0916  |   3.4571  |       0.2489346       |         0.2623        |
| 2048x2048 |146.8297029|27.4458059 |27.0261917 |      1.8790952        |       1.9128891       |
| 4096x4096 |1457.051280|215.775593 |217.0205408|      14.3551922       |       15.1342266      |
