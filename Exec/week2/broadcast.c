/* broadcast.c
 * Broadcast and integer from process 0 to all processes with rank != 0 
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
  MPI_Status  status;        /* status for receive   */
  int broadcast_integer; /*integer for broadcasting */
  int spacing; /*distance between sending processes*/
  int stage; /* stage of algorithm */
 
  /* Start up MPI */
  MPI_Init(&argc, &argv);
  
  /* Find out process rank  */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  
  /* Find out number of processes */
  MPI_Comm_size(MPI_COMM_WORLD, &p);
 
 broadcast_integer=-1;
 if (my_rank == 0) broadcast_integer=100;


  spacing=p;
  stage=0;
  while (spacing>1){
    if (my_rank % spacing == 0) 
      {
        dest = my_rank+spacing/2;
        printf("Process %d, sending message to process %d at stage %d \n",my_rank,dest,stage);
        MPI_Send(&broadcast_integer, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
      } 

    else  if (my_rank % (spacing/2) == 0)
      { 
        source=my_rank-spacing/2;
        printf("Process %d, receive message from process %d at stage %d \n",my_rank,source,stage);
        MPI_Recv(&broadcast_integer, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status); 
      }

//uncomment for debugging
//sleep(1);
//MPI_Barrier(MPI_COMM_WORLD);

    spacing=spacing/2;
    stage=stage+1;

  }
  
  /* Shut down MPI */
  MPI_Finalize();

  printf("Process %d has integer %d \n",my_rank,broadcast_integer);



  return 0;
} 
