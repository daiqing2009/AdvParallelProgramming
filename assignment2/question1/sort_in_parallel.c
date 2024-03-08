/*
 * Sort random number in parallel
 * 1. Genereate random number in each processs independently
 * 2. Communicate to related process according to the value of numbers
 * 3. Sort within each process
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"

#define INIT_MSG_SIZE 67108864
#define MAX_ITER 10

int main(int argc, char *argv[])
{
    int procNum, rank, next, prev, tag1 = 0, tag2 = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request reqs[procNum - 1];
    MPI_Status stats[procNum - 1];
    /* TODO: Read the input of N */

    /* genarate Random Number array */
    srand(rank);

    /* dispatch number to corresponding process: 
    process 0 has the random numbers between 0 and 1/p, 
    process 1 has numbers between 1/p and 2/p, and so on until 
    the last process has numbers between (p-1)/p and 1 */

    /* sort on recieved batch of number via insert sort algorithm asynchronizedly */
    
    if (rank == 0)
        prev = numtasks - 1;
    if (rank == numtasks - 1)
        next = 0;

    int msgSize = INIT_MSG_SIZE;
    for (int iter = 0; iter < MAX_ITER; iter++)
    {
        tag1 = iter;
        tag2 = iter;
        // printf("rank %d before msgSize\n", rank);
        char *recPre = (char *)malloc(msgSize * sizeof(char));
        char *recNext = (char *)malloc(msgSize * sizeof(char));
        char *sendPre = (char *)malloc(msgSize * sizeof(char));
        char *sendNext = (char *)malloc(msgSize * sizeof(char));
        // printf("rank %d after msgSize\n", rank);
        // make sure different iter will be recieved by corresponding buffer
        int i = 0;
        for (i = 0; i < msgSize; i++)
        {
            sendPre[i] = prev + '0';
            sendNext[i] = next + '0';
        }
        printf("proc(%d):sendPre=%.10s... & sendNext = %.10s...\n", rank, sendPre, sendNext);

        MPI_Irecv(recPre, msgSize, MPI_CHAR, prev, tag1,
                  MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(recNext, msgSize, MPI_CHAR, next, tag2,
                  MPI_COMM_WORLD, &reqs[1]);

        MPI_Isend(sendPre, msgSize, MPI_CHAR, prev, tag2,
                  MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(sendNext, msgSize, MPI_CHAR, next, tag1,
                  MPI_COMM_WORLD, &reqs[3]);

        MPI_Waitall(4, reqs, stats);

        printf("Proc(%d) communicated proc(%d) with tag(%d) and proc(%d) with tag(%d) of msgSize(%d) \n",
               rank, prev, tag2, next, tag1, msgSize);
        printf("proc(%d):recPre=%.10s... & recNext = %.10s...\n", rank, recPre, recNext);

        free(recPre);
        free(recNext);
        free(sendPre);
        free(sendNext);

        msgSize = msgSize * 2;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
