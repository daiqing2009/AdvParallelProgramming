/* getdata.c */
#include "stdio.h"
#include "mpi.h"

void Get_data( float* a_ptr, float* b_ptr, int* n_ptr, 
	       int my_rank, int p )
{ 
    int source = 0, dest, tag;
    MPI_Status status;
  
    if (my_rank == 0)
    {
        printf("Rank %d: Enter a, b, and n\n", my_rank);
        scanf("%f %f %d", a_ptr, b_ptr, n_ptr);  
      
        for (dest = 1; dest < p; dest++)
	{
            tag = 0;
            MPI_Send(a_ptr, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
            tag = 1;
            MPI_Send(b_ptr, 1, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
            tag = 2;
            MPI_Send(n_ptr, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
        }
    } 
    else 
    {
        tag = 0;
        MPI_Recv(a_ptr, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
        tag = 1;
        MPI_Recv(b_ptr, 1, MPI_FLOAT, source, tag, MPI_COMM_WORLD, &status);
        tag = 2;
        MPI_Recv(n_ptr, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
    }
} 
