/*
Implementation of SAXPY accelerated with CUDA.

A CPU implementation is also included for comparison.

No timing calls or error checks in this version, for clarity.

Compile on graham with:

nvcc -arch=sm_60 -O2 saxpy_cuda.cu

nvprof ./a.out


*/

#include "cuda.h" /* CUDA runtime API */
#include "cstdio"

#define SQRT          1024
#define BLOCK_SIZE_X  32
#define BLOCK_SIZE_Y  16

void saxpy_cpu(float *vecY, float *vecX, float alpha, int n)
{
  int i;

  for (i = 0; i < n; i++)
    vecY[i] = alpha * vecX[i] + vecY[i];
}

__global__ void saxpy_gpu(float *vecY, float *vecX, float alpha, int n)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int i = x + y * blockDim.x * gridDim.x;
  if (i < n)
    vecY[i] = alpha * vecX[i] + vecY[i];
}

int main(int argc, char *argv[])
{
  float *x_host, *y_host; /* arrays for computation on host*/
  float *x_dev, *y_dev;   /* arrays for computation on device */
  float *y_shadow;        /* host-side copy of device results */

  int n = SQRT * SQRT;
  float alpha = 0.5f;
  int nerror;

  size_t memsize;
  int i;

  memsize = n * sizeof(float);

  /* allocate arrays on host */

  x_host = (float *)malloc(memsize);
  y_host = (float *)malloc(memsize);
  y_shadow = (float *)malloc(memsize);

  /* allocate arrays on device */

  cudaMalloc((void **)&x_dev, memsize);
  cudaMalloc((void **)&y_dev, memsize);

  /* catch any errors */

  /* initialize arrays on host */

  for (i = 0; i < n; i++)
  {
    x_host[i] = rand() / (float)RAND_MAX;
    y_host[i] = rand() / (float)RAND_MAX;
  }

  /* copy arrays to device memory (synchronous) */

  cudaMemcpy(x_dev, x_host, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(y_dev, y_host, memsize, cudaMemcpyHostToDevice);

  /* set up device execution configuration */
  // blockSize = 512;
  // nBlocks = n / blockSize + (n % blockSize > 0);
  dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  dim3 gridSize(SQRT / BLOCK_SIZE_X, SQRT / BLOCK_SIZE_Y);

  /* execute kernel (asynchronous!) */
  saxpy_gpu<<<gridSize, blockSize>>>(y_dev, x_dev, alpha, n);

  /* execute host version (i.e. baseline reference results) */
  saxpy_cpu(y_host, x_host, alpha, n);

  /* retrieve results from device (synchronous) */
  cudaMemcpy(y_shadow, y_dev, memsize, cudaMemcpyDeviceToHost);

  /* guarantee synchronization */
  cudaDeviceSynchronize();

  /* check results */
  nerror = 0;
  for (i = 0; i < n; i++)
  {
    if (y_shadow[i] != y_host[i])
      nerror = nerror + 1;
  }
  printf("test comparison shows %d errors\n", nerror);

  /* free memory */
  cudaFree(x_dev);
  cudaFree(y_dev);
  free(x_host);
  free(y_host);
  free(y_shadow);

  return 0;
}
