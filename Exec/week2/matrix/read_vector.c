/* read_vector.c */
#include <stdio.h>
#include "mpi.h"
#include "matrix_vector.h"

void Read_vector(char *prompt, float local_x[], int local_n,
        int my_rank, int p)
{
    int   i;
    //TODO: change to lenght of n: local_n * p
    float temp_vector[MAX_ORDER];

    if (my_rank == 0) 
    {
        printf("%s\n", prompt);
        for (i = 0; i < p*local_n; i++) 
            scanf("%f", &temp_vector[i]);
    }

    MPI_Scatter(temp_vector, local_n, MPI_FLOAT, 
            local_x, local_n, MPI_FLOAT,
            0, MPI_COMM_WORLD);

} 
