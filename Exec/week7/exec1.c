/* copy answer from Het Thumar */
#include <stdio.h>
#include <omp.h>

#define SIZE 5

double dot_product(double *a, double *b, int size)
{
    double result = 0.0;

#pragma omp parallel for reduction(+ : result)
    for (int i = 0; i < size; ++i)
    {
        result += a[i] * b[i];
    }

    return result;
}

int main()
{
    double vec1[SIZE] = {9.0, 8.0, 7.0, 6.0, 5.0};
    double vec2[SIZE] = {4.0, 5.0, 2.0, 9.0, 7.0};

    double result = dot_product(vec1, vec2, SIZE);
    printf("Dot product: %.2f\n", result);
    return 0;
}