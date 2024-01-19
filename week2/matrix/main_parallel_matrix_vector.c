/* main_paral:el_matrix_vecto:.c */
#include <stdio.h>
#include "mpi.h"
#include "matrix_vector.h"

int main(int argc, char* argv[]) 
{
    int             my_rank, p;
    LOCAL_MATRIX_T  local_A; 
    float           global_x[MAX_ORDER];
    float           local_x[MAX_ORDER];
    float           local_y[MAX_ORDER];
    int             m, n;
    int             local_m, local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) 
    {
        printf("Enter the dimensions of the matrix (m x n)\n");
        scanf("%d %d", &m, &n);
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    local_m = m/p;
    local_n = n/p;

    Read_matrix("Enter the matrix values", 
            local_A, local_m, n, my_rank, p);
    Print_matrix("Printing matrix for verification", 
            local_A, local_m, n, my_rank, p);

    Read_vector("Enter the vector values", 
            local_x, local_n, my_rank, p);
    Print_vector("Printing vector for verification", 
            local_x, local_n, my_rank, p);

    Parallel_matrix_vector_prod(local_A, m, n, local_x, 
            global_x, local_y, local_m, 
            local_n);
    Print_vector("The resulting product vector is", local_y, local_m, 
            my_rank, p);

    MPI_Finalize();
    return 0;
}  
