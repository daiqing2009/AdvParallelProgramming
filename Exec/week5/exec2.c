#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

int main(int argc, char *argv[])
{

    int my_rank; /* rank of process      */
    int p;       /* number of processes  */
    MPI_Comm *communicators;
    int cm;
    int color;
    int key;

    int number_being_sent; // this plays the role of message in this program
    int source, dest;
    int tag = 0;

    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* Find out process rank  */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    communicators = (MPI_Comm *)malloc((p - 1) * sizeof(MPI_Comm));

    // create communicator for each pair of processes doing send->recv between each other
    for (cm = 0; cm < p - 1; cm++)
    {
        color = my_rank;
        key = my_rank;

        // picking this arbitrary number to give distinct color to two processes involved in communications
        if (cm == my_rank)
            color = p + 12345;
        if (cm == my_rank - 1)
            color = p + 12345;

        MPI_Comm_split(MPI_COMM_WORLD, color, key, &communicators[cm]);
    }
    

    if (my_rank == 0)
    {
        number_being_sent = 999; // this is our message
        printf("number being sent on process 0 is %d \n", number_being_sent);
        // note that in the new communicator used here there are only 2 processes,hence destination always 1,source always 0
        // it was important to use the key in MPI_Comm_split to preserve process order
        dest = 1;
        MPI_Send(&number_being_sent, 1, MPI_INT, dest, tag, communicators[my_rank]);
    }
    else if (my_rank == p - 1)
    {
        source = 0;
        MPI_Recv(&number_being_sent, 1, MPI_INT, source, tag, communicators[my_rank - 1], MPI_STATUS_IGNORE);
        printf("number received on process p-1 is %d \n", number_being_sent);
    }
    else
    {
        source = 0;
        MPI_Recv(&number_being_sent, 1, MPI_INT, source, tag, communicators[my_rank - 1], MPI_STATUS_IGNORE);
        printf("number received on prod(%d) is %d \n", my_rank, number_being_sent);
        dest = 1;
        MPI_Send(&number_being_sent, 1, MPI_INT, dest, tag, communicators[my_rank]);
        printf("number sent from prod(%d) is %d \n", my_rank, number_being_sent);
    }

    /* Shut down MPI */
    MPI_Finalize();

    free(communicators);
    return 0;
}
