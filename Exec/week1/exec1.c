/* greetings.c
 * Send a message from all processes with rank != 0
 * to process 0.
 * Process 0 prints the messages received.

 Compile and run:

 mpicc greetings.c -o greetings.x
 mpirun -np 4 ./greetings.x


 */
#include "stdio.h"
#include "string.h"
#include "mpi.h"

int main(int argc, char* argv[])
{
	int         my_rank;       /* rank of process      */
	int         p;             /* number of processes  */
	int         source;        /* rank of sender       */
	int         dest;          /* rank of receiver     */
	int         tag = 0;       /* tag for messages     */
	char        message[100];  /* storage for message  */
	MPI_Status  status;        /* status for receive   */

	/* Start up MPI */
	MPI_Init(&argc, &argv);

	/* Find out process rank  */
	MPI_Comm_rank(MPI_COMM{
		/* Create message */
		dest = my_rank < p-1? my_rank+1 :0;	
		sprintf(message, "Greetings from process %d to %d!",
				my_rank,dest);
		/* Use strlen+1 so that '\0' gets transmitted */
//		MPI_Send(message, strlen(message)+1, MPI_CHAR,
//				dest, tag, MPI_COMM_WORLD);
        MPI_Send(message, strlen(message)+1, MPI_CHAR,
                0, tag, MPI_COMM_WORLD);
	}
	else
	{ /* my_rank == 0 */
		for (source = 1; source < p; source++)
		{
			MPI_Recv(message, 100, MPI_CHAR, source, tag,
					MPI_COMM_WORLD, &status);
			printf("%s\n", message);
		}
	}

	/* Shut down MPI */
	MPI_Finalize();

    return 0;
}
