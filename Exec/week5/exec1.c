/* Consider N processes arranged in a row.
* Create communicators for each consecutive pair,
i.e. process 0 and 1, process 1 and 2, up to process N-2 and N-1.
Then send information from process 0 to process N-1 in a series of messages
using only those communicators, without using MPI_COMM_WORLD.
 * Input: none
 * Output:  Results of doing a broadcast across each of
            the q communicators.
 * Note:  Assumes the number of processes, p = q^2
 * Compile with: mpicc comm_split.c  -lm
 */
#include "stdio.h"
#include "string.h"
#include "mpi.h"

#define MSG_LEN 10

int main(int argc, char *argv[])
{
    int p, my_rank;
    int source, dest;
    int tag = 0;
    int i = 0;
    int ranks[2] = {0, 1};
    char message[10]="PENDING";
    MPI_Group orig_group;
    MPI_Group group_chain[2];
    MPI_Comm comm_chain[2];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // TODO: check num of p is greater than 2
    printf("program ready for proc(%d)\n", my_rank);
    /* buid up the comm chain */
    MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

    if (my_rank > 0)
    {
        ranks[0] = my_rank - 1;
        ranks[1] = my_rank;
        MPI_Group_incl(orig_group, 2, ranks, &group_chain[0]);
        MPI_Comm_create(MPI_COMM_WORLD, group_chain[0], &comm_chain[0]);
        printf("former comm_chain of proc(%d) created.\n", my_rank);
    }
    if (my_rank < p - 1)
    {
        ranks[0] = my_rank;
        ranks[1] = my_rank + 1;
        MPI_Group_incl(orig_group, 2, ranks, &group_chain[1]);
        MPI_Comm_create(MPI_COMM_WORLD, group_chain[1], &comm_chain[1]);
        printf("latter comm_chain of proc(%d) created.\n", my_rank);
    }

    /* transmit the messages along with the chain*/
    if (my_rank > 0)
    {
        source = 0;
        MPI_Recv(message, MSG_LEN, MPI_CHAR, source, tag, comm_chain[0], MPI_STATUS_IGNORE);
        printf("proc(%d) recieved message (%.5s) via local comm\n", my_rank, message);
    }

    if (my_rank < p - 1)
    {
        if (my_rank == 0)
        {
            sprintf(message, "DONE");
        }
        dest = 1;
        MPI_Send(message, MSG_LEN, MPI_CHAR, dest, tag, comm_chain[1]);
        printf("proc(%d) sent message (%.5s) via local comm\n", my_rank, message);
    }

    MPI_Finalize();
    return 0;
}
