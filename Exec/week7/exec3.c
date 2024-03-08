/* copy from the answer from Vaibhav Kaushal*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double sumArrayExplicitly(double *inputArray, int arrayLength)
{
    double totalSum = 0.0;
    double sumLocal = 0.0;
    for (int index = 0; index < arrayLength; ++index)
    {
        sumLocal += inputArray[index];
    }
    totalSum += sumLocal;

    return totalSum;
}

double sumArrayWithPragma(double *inputArray, int arrayLength)
{
    double totalSum = 0.0;
    for (int index = 0; index < arrayLength; ++index)
    {
        totalSum += inputArray[index];
    }
    return totalSum;
}

int main()
{
    const int dataSize = 34657800;
    double *dataArray = (double *)malloc(dataSize * sizeof(double));
    for (int i = 0; i < dataSize; ++i)
    {
        dataArray[i] = 1.0;
    }

    double startTimeExplicit = omp_get_wtime();
    double sumResultExplicit = sumArrayExplicitly(dataArray, dataSize);
    double endTimeExplicit = omp_get_wtime();

    double startTimePragma = omp_get_wtime();
    double sumResultPragma = sumArrayWithPragma(dataArray, dataSize);
    double endTimePragma = omp_get_wtime();

    printf("Sum with Explicit Function: %f\n", sumResultExplicit);
    printf("Time for Explicit Sum: %f seconds\n", endTimeExplicit - startTimeExplicit);
    printf("\nSum with Pragma Placeholder: %f\n", sumResultPragma);
    printf("Time for Pragma Placeholder Sum: %f seconds\n", endTimePragma - startTimePragma);

    free(dataArray);
    return 0;
}