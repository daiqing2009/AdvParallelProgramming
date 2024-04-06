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

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 256

// Total number of threads (total number of elements to process in the kernel):
// #define INIT_NMAX
#define NMAX 92672

// Number of times to run the test (for scaling of dataset):
#define NTESTS 3

// Maximum value of distance
#define MAX_DIST 1.42

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define SQUARE(X) ((X) * (X))

// Input array (global host memory):
float h_X[NMAX];
float h_Y[NMAX];

__device__ float d_X[NMAX];
__device__ float d_Y[NMAX];
__device__ float d_min_k1;
__device__ float d_min_k2;

// It messes up with y!
int timeval_subtract(double *result, struct timeval *x, struct timeval *y);

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static inline __device__ float fatomicMin(float *addr, float value)
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
    d_min_k1 = MAX_DIST;
    d_min_k2 = MAX_DIST;

    return;
}

// one tread per particle
__global__ void OneThreadPerParticleKernel()
{

    __shared__ float b_min[BLOCK_SIZE];

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float thread_min = MAX_DIST, next_dist = MAX_DIST;

    // Not needed, because NMAX is a power of two:
    // if (i >= NMAX)
    //     return;

    // calculate the distance related to i and find the min of current thread
    for (int j = i + 1; j < NMAX; j++)
    {
        next_dist = sqrt(SQUARE(d_X[i] - d_X[j]) + SQUARE(d_Y[i] - d_Y[j]));
        thread_min = (next_dist < thread_min) ? next_dist : thread_min;
    }
    b_min[threadIdx.x] = thread_min;

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
        fatomicMin(&d_min_k1, b_min[0]);
    }

    return;
}

__global__ void OneThreadPerPairKernel()
{
    __shared__ float b_min[BLOCK_SIZE];

    int k = threadIdx.x + blockDim.x * blockIdx.x;
    // find corresponding row and column in the upper diagonal
    /** A visual digram of 8 particles , row is i indexed while column is j indexed,
     *      0   1   2   3   4   5   6   7
     * 0    X   A   A   A   A   A   A   A
     * 1        X   B   B   B   B   B   B
     * 2            X   C   C   C   C   C
     * 3                X   D   D   D   D
     * 4                    X   D   D   D
     * 5                        X   C   C
     * 6                            X   B
     * 7                                X
     */
    // row is staring row and col of each series (A, B, C, D etc.)
    int row = k / (NMAX - 1);
    int col = row + 1 + (k % (NMAX - 1));

    int i = (col > NMAX - 1) ? NMAX - 1 - row : row;
    int j = (col > NMAX - 1) ? col - row : col;

    // calculate the distance related to i and find the min of current thread
    b_min[threadIdx.x] = sqrt(SQUARE(d_X[i] - d_X[j]) + SQUARE(d_Y[i] - d_Y[j]));

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
        fatomicMin(&d_min_k2, b_min[0]);
    }

    return;
}

int main(int argc, char **argv)
{
    struct timeval tdr0, tdr1, tdr2, tdr3, tdr4, tdr5;
    double cputime, k1time, k2time;
    double min0;
    float next_dist, min_k1, min_k2;
    int error;
    // int NMAX;
    int NBLOCKS;

    // Loop to run the timing test multiple times:
    for (int kk = 0; kk < NTESTS; kk++)
    {
        // Initializing random number generator:
        srand(kk + 1);

        // Initializing the input array:
        for (int i = 0; i < NMAX; i++)
        {
            h_X[i] = (float)rand() / (float)RAND_MAX;
            h_Y[i] = (float)rand() / (float)RAND_MAX;
        }

        // print first few particles
        for (int i = 0; i < 3; i++)
            printf("No %d of particle pair: (%.4f, %.4f)\n", i, h_X[i], h_Y[i]);

        // Computer distances in a CPU serial function
        gettimeofday(&tdr0, NULL);

        // Find the minimal in serial way
        min0 = MAX_DIST;
        next_dist = MAX_DIST;
        for (int i = 0; i < NMAX; i++)
        {
            // only need to calculate the upper half of the diagnal matrix
            for (int j = i + 1; j < NMAX; j++)
            {
                next_dist = sqrt(SQUARE(h_X[i] - h_X[j]) + SQUARE(h_Y[i] - h_Y[j]));
                min0 = (next_dist < min0) ? next_dist : min0;
            }
        }

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
        NBLOCKS = NMAX / BLOCK_SIZE;
        OneThreadPerParticleKernel<<<NBLOCKS, BLOCK_SIZE>>>();

        gettimeofday(&tdr3, NULL);

        // Copying the result back to host (we time it):
        if (error = cudaMemcpyFromSymbol(&min_k1, d_min_k1, sizeof(float), 0, cudaMemcpyDeviceToHost))
        {
            printf("Error copy from min_k1 %d\n", error);
            exit(error);
        }

        if (error = cudaDeviceSynchronize())
        {
            printf("Error cudaDeviceSynchronize for min_k1 %d\n", error);
            exit(error);
        }

        gettimeofday(&tdr4, NULL);

        // Hybrid binary/atomic reduction:
        NBLOCKS = NMAX / BLOCK_SIZE / 2 * (NMAX - 1);
        OneThreadPerPairKernel<<<NBLOCKS, BLOCK_SIZE>>>();

        gettimeofday(&tdr5, NULL);

        // Copying the result back to host (we time it):
        if (error = cudaMemcpyFromSymbol(&min_k2, d_min_k2, sizeof(float), 0, cudaMemcpyDeviceToHost))
        {
            printf("Error copy from min_k2 %d\n", error);
            exit(error);
        }

        if (error = cudaDeviceSynchronize())
        {
            printf("Error cudaDeviceSynchronize for min_k2 %d\n", error);
            exit(error);
        }

        timeval_subtract(&cputime, &tdr1, &tdr0);
        printf("CPU Time: %e\n", cputime);

        printf("Min distance between %d particle pair: GPU k1(%.7f) vs CPU(%.7f) (relative error %e)\n", NMAX, min_k1, min0, fabs((double)min_k1 - min0) / min0);
        timeval_subtract(&k1time, &tdr3, &tdr2);
        printf("GPU kernel1(OneThreadPerParticle) Time: %e\n", k1time);

        printf("Min distance between %d particle pair: GPU k2(%.7f) vs CPU(%.7f) (relative error %e)\n", NMAX, min_k2, min0, fabs((double)min_k2 - min0) / min0);
        timeval_subtract(&k2time, &tdr5, &tdr4);
        printf("GPU kernel2(OneThreadPerPair) Time: %e\n", k2time);

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