/* a parallel program which has each process generate N random values of (x,y)
 * evaluate the Eggholder function on these values, and then finds the lowest value across all processes
 * at the end printing this lowest value and the corresponding values of x and y. *
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define N 1000 /* the number of iteration each process should try */
#define SCALE 512 /* the scale of random parameters */

extern float Eggholder(float x, float y);

int main(int argc, char** argv)
{
    int my_rank; /* rank of process */
    int p; /* number of processes */
    int source; /* rank of sender */
    int dest; /* rank of receiver */
    MPI_Status status; /* status for receive */

    int n =0; /* the trail of finding local_min of each process */
    float x, y = 0.0; /* the value of init */
    float result, local_min, global_min = 1024.0; /* the init of minimums of eggholder function, maxium of possible function */

    /* Start up MPI */
    MPI_Init(&argc, &argv);
    /* Find out process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    /* Find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* seed the random value */
    srand(my_rank);

    /* calculate the local minimum of each process */
    for(int i=0; i <N; i++){
        /* initialize the random value of x and y */
        x = ((float)rand()/(float)(RAND_MAX)) * SCALE;
        y = ((float)rand()/(float)(RAND_MAX)) * SCALE;

        result = Eggholder(x,y);
        if(result<local_min){
            local_min = result;
        }
    }

    printf("the local minimun found of process(%d) is %f\n",my_rank, local_min );

    /* find global_min of local_min calculated by each process */
    MPI_Reduce(&local_min, &global_min, 1, MPI_FLOAT,
            MPI_MIN, 0, MPI_COMM_WORLD);

    if(my_rank == 0)
    {
        printf("the global minimum found of eggholder function is %f with x =  %f and y = %f\n",
                global_min, x, y);
    }
    /* Shut down MPI */
    MPI_Finalize();
    return 0;
}


