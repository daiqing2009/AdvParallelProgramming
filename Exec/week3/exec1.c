/* Test for Deadlock
 * write an unsafe program running on two processes,
 * where both do an MPI_Send followed by an MPI_Recv.
 * Then gradually increase the size of the message to see
 * how big it must become before the program runs out of buffer space and deadlocks.

 Compile and run:

 mpicc greetings.c -o greetings.x
 mpirun -np 4 ./greetings.x


*/
#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

#define INIT_MSG_SIZE 16
#define MAX_ITER 10

int main(int argc, char *argv[])
{
    int my_rank;       /* rank of process      */
    int p;             /* number of processes  */
    int source;        /* rank of sender       */
    int dest;          /* rank of receiver     */
    int tag = 0;       /* tag for messages     */
    MPI_Status status; /* status for receive   */

    /* Start up MPI */
    MPI_Init(&argc, &argv);

    /* Find out process rank  */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (p != 2)
    {
        printf("Aborted! Test for Deadlock runs on process num of 2");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int msgSize = INIT_MSG_SIZE * sizeof(int);

    for (int iter = 1; iter < MAX_ITER; iter *= 2)
    {
        msgSize = iter * INIT_MSG_SIZE * sizeof(int);
        int *message = (int*) malloc(msgSize);

        /* toogle the process ID since only 2 of process */
        source = my_rank % 2;
        dest = source;

        printf("Sending message of size(%d) from proc(%d) to proc(%d)\n", msgSize, my_rank, dest );
        MPI_Send(message, msgSize, MPI_INT, dest, tag, MPI_COMM_WORLD);
        MPI_Recv(message, msgSize, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        printf("Message of size(%d) recieved from proc(%d) to proc(%d)\n", msgSize, source, my_rank );

        free(message);
    }

    /* Shut down MPI */
    MPI_Finalize();

    return EXIT_SUCCESS;
}
