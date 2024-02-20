/* print_matrix.c */
#include <stdio.h>
#include "mpi.h"
#include "matrix_vector.h"

void Print_matrix(char *title, LOCAL_MATRIX_T  local_A,
        int local_m, int n, int my_rank, int p)
{
    int   i, j;
    //TODO: set matrix to size of m * n
    float temp_matrix[MAX_ORDER][MAX_ORDER];

    MPI_Gather(local_A, local_m*MAX_ORDER, MPI_FLOAT, 
            temp_matrix, local_m*MAX_ORDER, MPI_FLOAT, 
            0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("%s\n", title);
        for (i = 0; i < p*local_m; i++) 
        {
            for (j = 0; j < n; j++)
                printf("%4.1f ", temp_matrix[i][j]);
            printf("\n");
        }
    } 
} 
