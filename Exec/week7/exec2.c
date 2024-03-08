/* copy from answer from @Het Thumar */
#include <stdio.h>
#include <omp.h>

// Prime number function
int is_prime(int n)
{
    if (n <= 1)
        return 0;
    if (n <= 3)
        return 1;
    if (n % 2 == 0 || n % 3 == 0)
        return 0;
    for (int i = 5; i * i <= n; i += 6)
    {
        if (n % i == 0 || n % (i + 2) == 0)
            return 0;
    }
    return 1;
}

int main()
{
    // setting the bound to 1000000
    int bound = 1000000;
    int prime_count = 0;
    printf("Different scheduling methods\n");

    printf("static scheduling:\n");
    double start_time = omp_get_wtime();
#pragma omp parallel for schedule(static) reduction(+ : prime_count)
    for (int i = 2; i <= bound; i++)
    {
        if (is_prime(i))
        {
            prime_count++;
        }
    }
    double end_time = omp_get_wtime();

    printf("Total number of prime numbers found here: %d\n", prime_count);
    printf("Time taken for static scheduling: %f seconds\n", end_time - start_time);

    prime_count = 0;

    printf("\ndynamic scheduling:\n");
    start_time = omp_get_wtime();
#pragma omp parallel for schedule(dynamic) reduction(+ : prime_count)
    for (int i = 2; i <= bound; i++)
    {
        if (is_prime(i))
        {
            prime_count++;
        }
    }
    end_time = omp_get_wtime();
    printf("Total number of prime numbers found here: %d\n", prime_count);
    printf("Time taken for dynamic scheduling: %f seconds\n", end_time - start_time);

    prime_count = 0;

    printf("\nguided scheduling:\n");
    start_time = omp_get_wtime();
#pragma omp parallel for schedule(guided) reduction(+ : prime_count)
    for (int i = 2; i <= bound; i++)
    {
        if (is_prime(i))
        {
            prime_count++;
        }
    }
    end_time = omp_get_wtime();
    printf("Total number of prime numbers found here: %d\n", prime_count);
    printf("Time taken for guided scheduling: %f seconds\n", end_time - start_time);

    return 0;
}