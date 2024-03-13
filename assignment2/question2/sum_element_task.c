#include <stdio.h>
#include <omp.h>
#include <stdbool.h>

#define PRT_PRE 3
#define PRT_POST 2
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

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
    for (ip = 0; ip < m; ip++)
    {
        for (jp = 0; jp < n; jp++)
        {
            if (jp == 0)
            {
                b[ip * m + jp] = b[(ip - 1) * m + jp] + a[ip * m + jp];
            }
            else if (ip == 0)
            {
                b[ip * m + jp] = b[ip * m + jp - 1] + a[ip * m + jp];
            }
            else
            {
                b[ip * m + jp] = b[(ip - 1) * m + jp] + b[ip * m + jp - 1] + a[ip * m + jp];
            }
        }
    }

    print_matrix("Matrix B:", b, m, n);

    return 0;
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