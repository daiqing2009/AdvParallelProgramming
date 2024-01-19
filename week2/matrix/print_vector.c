/* print_vector.c */
#include <stdio.h>
#include "mpi.h"
#include "matrix_vector.h"

void Print_vector(char *title, float  local_y[] ,
        int local_m, int my_rank, 
        int p)
{
    int   i;
    float temp_vector[MAX_ORDER];

    MPI_Gather(local_y, local_m, MPI_FLOAT, 
            temp_vector, local_m, MPI_FLOAT,
            0, MPI_COMM_WORLD);

    if (my_rank == 0) 
    {
        printf("%s\n", title);
        for (i = 0; i < p*local_m; i++)
            printf("%4.1f ", temp_vector[i]);
        printf("\n");
    } 
}  
