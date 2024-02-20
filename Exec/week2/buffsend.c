/* Program demonstrates sending data using buffer */

#include "stdio.h"
#include "stdlib.h"
#include "mpi.h"

#define N_DATA 100000

int main(argc,argv)
int argc;
char *argv[];
{

int     numtasks, rank, error, i, dest=1, tag=123, source=0, size;
double  data[N_DATA], result;
void    *buffer;

MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);

if (numtasks != 2) {
  printf("Must run with 2 tasks. Terminating\n");
  MPI_Finalize();
  }
printf ("Task %d starting...\n", rank);


/************************* Sending task ****************************/
if (rank == 0) {

  /* Initialize data */
  for(i=0; i<N_DATA; i++)
       data[i] = (double)random();

  /* Determine buffer size needed including any required MPI overhead */
  MPI_Pack_size (N_DATA, MPI_DOUBLE, MPI_COMM_WORLD, &size);
  size = size +  MPI_BSEND_OVERHEAD;
  printf("Using buffer size= %d\n",size);

  /* Attach buffer, do buffered send, detach buffer */
  buffer = (void*)malloc(size);
  error = MPI_Buffer_attach(buffer, size);
  if (error != MPI_SUCCESS) {
    printf("Failed to attach buffer. Error code= %d Terminating\n", error);
    MPI_Finalize();
    }
  MPI_Bsend(data, N_DATA, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
  printf("Task 0 has sent message \n");
  MPI_Buffer_detach(&buffer, &size);
  free (buffer);

  }


/************************** Receiving task ***************************/
if (rank == 1) {
  MPI_Recv(data, N_DATA, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
  printf("Task 1 has received message \n"); 
  }


MPI_Finalize();
}

