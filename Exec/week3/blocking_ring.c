/*
 * blocking program for ring communication, which can't proceed beyond 2024
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "mpi.h"

#define INIT_MSG_SIZE 64
#define MAX_ITER 10

int main(int argc, char *argv[])
{
  int numtasks, rank, next, prev, tag1 = 0, tag2 = 0;

  MPI_Request reqs[4];
  MPI_Status stats[4];

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (numtasks > 10)
  {
    printf("this program can only work with np less than 10");
    return EXIT_FAILURE;
  }
  prev = rank - 1;
  next = rank + 1;

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
    char *recPre = (char *)malloc((msgSize+1) * sizeof(char));
    char *recNext = (char *)malloc((msgSize+1) * sizeof(char));
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

    MPI_Send(sendPre, msgSize, MPI_CHAR, prev, tag2,
             MPI_COMM_WORLD);
    MPI_Send(sendNext, msgSize, MPI_CHAR, next, tag1,
             MPI_COMM_WORLD);

    MPI_Recv(recPre, msgSize, MPI_CHAR, prev, tag1,
             MPI_COMM_WORLD, &stats[0]);
    MPI_Recv(recNext, msgSize, MPI_CHAR, next, tag2,
             MPI_COMM_WORLD, &stats[1]);
    // MPI_Waitall(4, reqs, stats);
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
