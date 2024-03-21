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
#include <math.h>
#include <sys/time.h>


// void saxpy_cpu(float *vecY, float *vecX, float alpha, int n) {
//     int i;

//     for (i = 0; i < n/2; i++)
//         vecY[i] = cos(vecX[i])
//     for(i=n/2;i<n)
// }

__global__ void saxpy_firstlast(float *vecY, float *vecX, int n)
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n / 2)
        vecY[i] = cos(vecX[i]);
    else
        vecY[i] = cos(vecX[i]);
}

__global__ void saxpy_evenodd(float *vecY, float *vecX, int n)
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i % 2 == 0)
        vecY[i] = cos(vecX[i]);
    else
        vecY[i] = cos(vecX[i]);
}

int
timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

int main(int argc, char *argv[])
{
    float *x_host, *y_host; /* arrays for computation on host*/
    float *x_dev, *y_dev;   /* arrays for computation on device */
    float *y_shadow;        /* host-side copy of device results */

    struct timeval tdr0, tdr1, tdr2, tdr;
    double firstlast_time, evenodd_time;

    int n = 1024 * 1024;
    // float alpha = 0.5f;
    // int nerror;

    size_t memsize;
    int i, blockSize, nBlocks;

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
    blockSize = 512;
    nBlocks = n / blockSize + (n % blockSize > 0);

    gettimeofday(&tdr0, NULL);

    /* execute kernel (asynchronous!) */
    saxpy_firstlast<<<nBlocks, blockSize>>>(y_dev, x_dev, n);

    /* retrieve results from device (synchronous) */
    cudaMemcpy(y_shadow, y_dev, memsize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    gettimeofday(&tdr1, NULL);
    tdr = tdr0;
    timeval_subtract(&firstlast_time, &tdr1, &tdr);

    /* execute kernel (asynchronous!) */
    saxpy_evenodd<<<nBlocks, blockSize>>>(y_dev, x_dev, n);

    /* retrieve results from device (synchronous) */
    cudaMemcpy(y_shadow, y_dev, memsize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&tdr2, NULL);
    tdr = tdr1;
    timeval_subtract(&evenodd_time, &tdr2, &tdr);

    printf ("Time for saxpy_firstlast: %e and Time for saxpy_evenodd: %e\n", firstlast_time, evenodd_time);

    /* execute host version (i.e. baseline reference results) */
    // saxpy_cpu(y_host, x_host, n);

    /* retrieve results from device (synchronous) */
    // cudaMemcpy(y_shadow, y_dev, memsize, cudaMemcpyDeviceToHost);

    /* guarantee synchronization */
    // cudaDeviceSynchronize();

    /* check results */
    // nerror=0;
    // for(i=0; i < n; i++) {
    //     if(y_shadow[i]!=y_host[i]) nerror=nerror+1;
    // }
    // printf("test comparison shows %d errors\n",nerror);

    /* free memory */
    cudaFree(x_dev);
    cudaFree(y_dev);
    free(x_host);
    free(y_host);
    free(y_shadow);

    return 0;
}
