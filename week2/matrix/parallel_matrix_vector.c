/* parallel_matrix_vector.c */
#include "mpi.h"
#include "matrix_vector.h"

    void Parallel_matrix_vector_prod
( LOCAL_MATRIX_T  local_A, int m, int n,
  float local_x[], float global_x[], float local_y[],
  int   local_m, int local_n)
{
    /* local_m = m/p, local_n = n/p */
    int i, j;

    MPI_Allgather(local_x, local_n, MPI_FLOAT,
            global_x, local_n, MPI_FLOAT,
            MPI_COMM_WORLD);

    for (i = 0; i < local_m; i++) 
    {
        local_y[i] = 0.0;
        for (j = 0; j < n; j++)
            local_y[i] = local_y[i] +
                local_A[i][j]*global_x[j];
    }
} 
