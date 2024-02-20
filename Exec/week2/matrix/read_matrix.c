/* read_matrix.c */
#include <stdio.h>
#include "mpi.h"
#include "matrix_vector.h"

void Read_matrix(char *prompt, LOCAL_MATRIX_T  local_A,
        int local_m, int n, int my_rank, 
        int p)
{
    int i, j;
    LOCAL_MATRIX_T  temp_matrix;

    /* Fill entries in temp_matrix with zeros, for subsequent overwrite */
    for (i = 0; i < p*local_m; i++)
        for (j = n; j < MAX_ORDER; j++)
            temp_matrix[i][j] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) 
    {
        printf("%s\n", prompt);
        for (i = 0; i < p*local_m; i++) 
            for (j = 0; j < n; j++)
                scanf("%f",&temp_matrix[i][j]);
    }
    MPI_Scatter(temp_matrix, local_m*MAX_ORDER, MPI_FLOAT, 
            local_A, local_m*MAX_ORDER, MPI_FLOAT, 
            0, MPI_COMM_WORLD);
}
