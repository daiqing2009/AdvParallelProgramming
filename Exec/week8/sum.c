#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define CHUNK_SIZE 1000

int sum(int *arr, int size)
{
    int result = 0;
    for (int i = 0; i < size; ++i)
    {
        result += arr[i];
    }
    return result;
}

int main()
{
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    // Dynamically allocate memory for the array
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL)
    {
        printf("Memory allocation failed.");
        return 1;
    }

    // Assign consecutive numbers to the array
    for (int index = 0; index < n; index++)
    {
        arr[index] = index + 1;
    }

    // Measure the start time
    double start_time = omp_get_wtime();

    int result = 0;
    #pragma omp parallel
    #pragma omp single
    {
        for (int i = 0; i < n; i += CHUNK_SIZE)
        {
            int chunk_size = (i + CHUNK_SIZE < n) ? CHUNK_SIZE : n - i;
            #pragma omp task firstprivate(i, chunk_size) shared(arr)
            {
                int local_sum = sum(&arr[i], chunk_size);
                #pragma omp atomic
                result += local_sum;
            }
        }
    }

    // Measure the end time
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    printf("The sum of the array elements is: %d\n", result);
    printf("Time taken: %.6f seconds\n", elapsed_time);

    // Free dynamically allocated memory
    free(arr);

    return 0;
}