#include <stdio.h>
#include <omp.h>
#include <stdbool.h>

#define PRT_PRE 3
#define PRT_POST 2
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

int constructB(int ip, int jp, int m, int n, int *a, int *b);

void print_matrix(char *prompt, int *mat, int m, int n);

int main()
{
    int n, m, ip, jp;
    int *a, *b;
    printf("Please enter array dimention(m * n): \n");
    scanf("%d %d", &m, &n);

    /* initialize array A */
    a = malloc(m * n * sizeof(int));
    for (ip = 0; ip < m; ip++)
    {
        for (jp = 0; jp < n; jp++)
        {
            a[ip * m + jp] = ip * m + jp;
        }
    }

    print_matrix("Matrix A:", a, m, n);
    /* contrauct B */
    b = malloc(m * n * sizeof(int));
    /* construct matrix B in a recursive way*/
    //     omp_set_dynamic(0);
    //     omp_set_num_threads(4);
    // #pragma omp parallel shared(m, n, a, b)
    // {
    constructB(m - 1, n - 1, m, n, a, b);
    // #pragma omp single
    print_matrix("Matrix B:", b, m, n);
    // }

    return 0;
}

int constructB(int ip, int jp, int m, int n, int *a, int *b)
{
    int last_row = -1, last_col= -1, value = 0;
    // #pragma opm task shared(last_row) firstprivate(ip, jp)
    if (ip > 0)
    {
        last_row = constructB(ip - 1, jp, m, n, a, b);
        value += last_row;
    }

    // #pragma opm task shared(last_col) firstprivate(ip, jp)
    if (jp > 0)
    {
        last_col = constructB(ip, jp - 1, m, n, a, b);
        value += last_col;
    }
    // #pragma omp taskwait
    value += a[ip * m + jp];
    if (ip > 0 && jp > 0)
        value -= a[(ip - 1) * m + jp - 1];

    b[ip * m + jp] = value;
    return value;
}

void print_matrix(char *prompt, int *mat, int m, int n)
{
    int i, j;
    printf("%s\n", prompt);
    // only print left-most portion of matrix when too large
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%d\t", mat[i * m + j]);
        }
        printf("\n");
    }
}