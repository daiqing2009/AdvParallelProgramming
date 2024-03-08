#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

int main(int argc, char *argv[])
{
    int my_rank; /* rank of process      */
    int p;       /* number of processes  */

    int ndims = 3;
    int dims[ndims];
    int periods[ndims];
    int reorder = 0;
    MPI_Comm grid_comm;
    int free_coords[3];
    MPI_Comm side_comm[3];
    int corner_number;
    int total_side[3];
    int i;
    int proc;

    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* Find out process rank  */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (p != 8)
    {
        printf("must run with exactly 8 processes \n");
        MPI_Finalize();
        return 0;
    }

    dims[0] = 2;
    dims[1] = 2;
    dims[2] = 2;

    periods[0] = 0;
    periods[1] = 1;
    periods[2] = 2;

    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &grid_comm);

    corner_number = my_rank; // the number held by each process is equal to its rank

    //  distribution of numbers on cube
    //     6-----7
    //   2-----3 |
    //   | |   | |
    //   | 4-----5
    //   0-----1
    // sides have sums (0+1+2+3),(4+5+6+7),(0+2+4+6),(1+3+5+7),(0+1+4+5),(2+3+6+7)
    //                 6        , 22      , 12      , 16      , 10      , 18

    for (i = 0; i < 3; i++)
    {
        free_coords[0] = 1;
        free_coords[1] = 1;
        free_coords[2] = 1;
        free_coords[i] = 0;
        MPI_Cart_sub(grid_comm, free_coords, &side_comm[i]);
        MPI_Allreduce(&corner_number, &total_side[i], 1, MPI_INT, MPI_SUM, side_comm[i]);
    }

    printf("process %d sum0=%d sum1=%d sum2=%d \n", my_rank, total_side[0], total_side[1], total_side[2]);

    /* Shut down MPI */
    MPI_Finalize();

    return 0;
}