#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define N (1 << 22) // Total number of elements in the array
#define THREADS_PER_BLOCK 256

// Kernel to access array elements with variable stride
__global__ void accessArray(float *arr, int stride)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < N; i += stride)
    {
        sum += arr[i];
    }

    // Ensure all threads have finished computation
    __syncthreads();

    // Only one thread per block writes the sum
    if (threadIdx.x == 0)
    {
        arr[blockIdx.x] = sum;
    }
}

int main()
{
    float *d_arr;
    float *h_arr = (float *)malloc(sizeof(float) * N);
    struct timeval start, end;

    // Initialize array with random values
    for (int i = 0; i < N; i++)
    {
        h_arr[i] = (float)rand() / RAND_MAX;
    }

    // Allocate memory on GPU
    cudaMalloc(&d_arr, sizeof(float) * N);
    cudaMemcpy(d_arr, h_arr, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Run the kernel with different stride values and measure time
    printf("Stride\tBandwidth (GB/s)\n");
    for (int stride = 1; stride <= N/2; stride *= 2)
    {
        gettimeofday(&start, NULL);

        accessArray<<<N / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_arr, stride);
        cudaDeviceSynchronize(); // Wait for all kernels to finish

        gettimeofday(&end, NULL);

        double elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0; // Convert to milliseconds
        elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;

        double dataSize = sizeof(float) * N;                                       // Total data size in bytes
        double bandwidth = dataSize / (elapsedTime * 1e-3) / (1024 * 1024 * 1024); // Bandwidth in GB/s

        printf("%d\t%f\n", stride, bandwidth);
    }

    // Free memory
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}