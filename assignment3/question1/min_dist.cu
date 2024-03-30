/* Atomic reduction solution.
 */

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 256

// Total number of threads (total number of elements to process in the kernel):
#define INIT_NMAX 4096

// Number of times to run the test (for scaling of dataset):
#define NTESTS 3

// Maximum value of
#define MAX_DIST 1.41

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define SQUARE(X) ((X) * (X))

#define NMAX INIT_NMAX * 2
// Input array (global host memory):
// float *h_X;
// float *h_Y;
// float *h_dist;
float h_X[NMAX];
float h_Y[NMAX];
float h_dist[NMAX * (NMAX - 1)];

__device__ float d_X[NMAX];
__device__ float d_Y[NMAX];
// __device__ float *d_dist;
__device__ float d_min;

// It messes up with y!
int timeval_subtract(double *result, struct timeval *x, struct timeval *y);

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
__device__ static float fatomicMin(float *addr, float value)
{
    float old = *addr, assumed;
    if (old <= value)
        return old;
    do
    {
        assumed = old;
        old = atomicCAS((unsigned int *)addr, __float_as_int(assumed), __float_as_int(value));
    } while (old != assumed);

    return old;
}

__global__ void init_kernel()
{
    d_min = MAX_DIST;
    return;
}

// one tread per particle
__global__ void OneThreadPerParticleKernel()
{

    __shared__ float b_min[BLOCK_SIZE];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float thread_min, next_dist = MAX_DIST;

    // Not needed, because NMAX is a power of two:
    // if (i >= NMAX)
    //     return;

    // calculate the distance related to i and find the min of current thread
    for (int j = i + 1; j < BLOCK_SIZE; j += 2)
    {
        next_dist = sqrt(SQUARE(d_X[i] - d_X[j]) + SQUARE(d_Y[i] - d_Y[j]));
        thread_min = (next_dist < thread_min) ? next_dist : thread_min;
    }

    // To make sure all threads finished calc
    __syncthreads();

    // find the min within the block
    int nTotalThreads = blockDim.x; // Total number of active threads;
    // only the first half of the threads will be active.
    while (nTotalThreads > 1)
    {
        int halfPoint = (nTotalThreads >> 1); // divide by two
        if (threadIdx.x < halfPoint)
        {
            int thread2 = threadIdx.x + halfPoint;
            b_min[threadIdx.x] = MIN(b_min[threadIdx.x], b_min[thread2]); // Pairwise summation
        }
        __syncthreads();
        nTotalThreads = halfPoint; // Reducing the binary tree size by two
    }

    // find the min among blocks, i.e. global min
    if (threadIdx.x == 0)
    {
        fatomicMin(&d_min, b_min[0]);
    }

    return;
}

__global__ void OneThreadPerPairKernel()
{
}

int main(int argc, char **argv)
{
    struct timeval tdr0, tdr1, tdr2, tdr3;
    double cputime, k1time, k2time;
    double min0;
    float min;
    int error;
    // int NMAX;
    int NBLOCKS;

    // Loop to run the timing test multiple times:
    for (int kk = 0; kk < NTESTS; kk++)
    {
        // double NMAX every time
        // NMAX = INIT_NMAX * (int)pow(2, kk);
        // printf("NMAX = %d", NMAX);

        NBLOCKS = NMAX * (NMAX - 1) / BLOCK_SIZE;

        // allocate CPU memory
        // h_X = (float *)malloc(NMAX * sizeof(float));
        // h_Y = (float *)malloc(NMAX * sizeof(float));

        // allocate GPU memory
        // cudaMalloc((void **)&d_X, NMAX * sizeof(float));
        // cudaMalloc((void **)&d_Y, NMAX * sizeof(float));
        // cudaMalloc((void **)&d_dist, NMAX * (NMAX - 1) * sizeof(float));

        // We don't initialize randoms, because we want to compare different strategies:
        // Initializing random number generator:
        srand(kk);

        // Initializing the input array:
        for (int i = 0; i < NMAX; i++)
        {
            h_X[i] = (float)rand() / (float)RAND_MAX;
            h_Y[i] = (float)rand() / (float)RAND_MAX;
        }

        // Computer distances in a CPU serial function
        gettimeofday(&tdr0, NULL);

        for (int i = 0; i > NMAX - 1; i++)
        {
            // only need to calculate the upper half of the diagnal matrix
            for (int j = 0; j < i; j++)
                h_dist[i * NMAX + j] = MAX_DIST;
            for (int j = i; j < NMAX; j++)
                h_dist[i * NMAX + j] = sqrt(SQUARE(h_X[i] - h_X[j]) + SQUARE(h_Y[i] - h_Y[j]));
        }

        // Find the minimal in serial way
        min0 = MAX_DIST;
        for (int i = 0; i < (NMAX - 1) * NMAX; i++)
            if (h_dist[i] < min0)
                min0 = (double)h_dist[i];

        gettimeofday(&tdr1, NULL);

        // Copying the data to device (we don't time it):
        if (error = cudaMemcpyToSymbol(d_X, h_X, NMAX * sizeof(float), 0, cudaMemcpyHostToDevice))
        {
            printf("Error copy X to device %d\n", error);
            exit(error);
        }
        if (error = cudaMemcpyToSymbol(d_Y, h_Y, NMAX * sizeof(float), 0, cudaMemcpyHostToDevice))
        {
            printf("Error copy Y to device %d\n", error);
            exit(error);
        }

        init_kernel<<<1, 1>>>();
        if (error = cudaDeviceSynchronize())
        {
            printf("Error %d\n", error);
            exit(error);
        }
        //--------------------------------------------------------------------------------
        gettimeofday(&tdr2, NULL);

        // Hybrid binary/atomic reduction:
        OneThreadPerParticleKernel<<<NBLOCKS, BLOCK_SIZE>>>();

        gettimeofday(&tdr3, NULL);
        // tdr = tdr0;
        // timeval_subtract(&restime, &tdr1, &tdr);

        // Copying the result back to host (we time it):
        if (error = cudaMemcpyFromSymbol(&min, d_min, sizeof(float), 0, cudaMemcpyDeviceToHost))
        {
            printf("Error %d\n", error);
            exit(error);
        }

        if (error = cudaDeviceSynchronize())
        {
            printf("Error %d\n", error);
            exit(error);
        }

        printf("Min: %e (relative error %e)\n", min, fabs((double)min - min0) / min0);

        timeval_subtract(&cputime, &tdr1, &tdr0);
        printf("CPU Time: %e\n", cputime);

        timeval_subtract(&k1time, &tdr3, &tdr2);
        printf("GPU kernel1(OneThreadPerParticle) Time: %e\n", k1time);
        // printf("GPU kernel1(OneThreadPerParticle) Time: %e\n", k2time);

        //--------------------------------------------------------------------------------
        // free GPU memory
        // cudaFree(d_X);
        // cudaFree(d_Y);
        // cudaFree(d_dist);

        // free CPU memory
        // free(h_X);
        // free(h_Y);
        // free(h_Y);
    } // kk loop

    return 0;
}

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(double *result, struct timeval *x, struct timeval *y)
{
    struct timeval result0;

    /* Perform the carry for the later subtraction by updating y. */
    if (x->tv_usec < y->tv_usec)
    {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
        y->tv_usec -= 1000000 * nsec;
        y->tv_sec += nsec;
    }
    if (x->tv_usec - y->tv_usec > 1000000)
    {
        int nsec = (y->tv_usec - x->tv_usec) / 1000000;
        y->tv_usec += 1000000 * nsec;
        y->tv_sec -= nsec;
    }

    /* Compute the time remaining to wait.
       tv_usec is certainly positive. */
    result0.tv_sec = x->tv_sec - y->tv_sec;
    result0.tv_usec = x->tv_usec - y->tv_usec;
    *result = ((double)result0.tv_usec) / 1e6 + (double)result0.tv_sec;

    /* Return 1 if result is negative. */
    return x->tv_sec < y->tv_sec;
}