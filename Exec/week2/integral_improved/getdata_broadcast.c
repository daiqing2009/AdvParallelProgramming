/* getdata_broadcast.c */
#include "stdio.h"
#include "mpi.h"

void Get_data(float* a_ptr, float* b_ptr, int* n_ptr,
	       int my_rank)
{
    if (my_rank == 0) 
    {
        printf("Enter a, b, and n\n");
        scanf("%f %f %d", a_ptr, b_ptr, n_ptr);
    }
    MPI_Bcast(a_ptr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_ptr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(n_ptr, 1, MPI_INT,   0, MPI_COMM_WORLD);
} 



