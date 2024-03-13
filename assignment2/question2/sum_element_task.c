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
            a[ip * n + jp] = ip * n + jp;
        }
    }

    // print_matrix("Matrix A:", a, m, n);
    /* contrauct B */
    b = malloc(m * n * sizeof(int));
    /* construct matrix B in a recursive way*/
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    #pragma omp parallel shared(m, n, a, b) 
    {
        constructB(m - 1, n - 1, m, n, a, b);
        // #pragma omp single
        // print_matrix("Matrix B:", b, m, n);
    }

    return 0;
}

int constructB(int ip, int jp, int m, int n, int *a, int *b)
{
    int last_row = 0, last_col = 0, diagnal =0, value = 0;
    if (ip > 0)
    {
        #pragma opm task shared(last_row) firstprivate(ip, jp)
        last_row = constructB(ip - 1, jp, m, n, a, b);
        #pragma omp taskwait
        value += last_row;
    }

    if (jp > 0)
    {
        #pragma opm task shared(last_col) firstprivate(ip, jp)
        last_col = constructB(ip, jp - 1, m, n, a, b);
        #pragma omp taskwait
        value += last_col;
    }
    value += a[ip * n + jp];
    // printf("B(%d,%d) = %d\n", ip, jp ,value);
    if (ip > 0 && jp > 0)
    {
        #pragma opm task shared(diagnal) firstprivate(ip, jp)
        // printf("ready to minute a(%d,%d) = %d\n", ip - 1, jp - 1 ,a[(ip - 1) * m + jp - 1]);
        diagnal = constructB(ip - 1, jp - 1, m, n, a, b);
        #pragma omp taskwait
        value -= diagnal;
    }

    b[ip * n + jp] = value;
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
            printf("%d\t", mat[i * n + j]);
        }
        printf("\n");
    }
}