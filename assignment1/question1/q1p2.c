/* a parallel program which has each process generate N random values of (x,y)
 * evaluate the Eggholder function on these values, and then finds the lowest value across all processes
 * at the end printing this lowest value and the corresponding values of x and y. *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <string.h>
#include <getopt.h>
#include "mpi.h"

extern float Eggholder(float x, float y);
extern char *optarg;
extern int opterr, optind;

#define N   100000 /*number counter to check time after N iterations*/
#define SEC 30 /* the nnumber of seconds processes should keep running */
#define SCALE 512 /* the scale of random parameters */
/* define and prompt the the parameter of program */
#define OPTSTR "vi:s:h"
#define USAGE_FMT  "%s [-v] [-s second to keep  process running] [-h]"
#define DEFAULT_PROGNAME "q1p2"

typedef struct {
    int verbose;
    int sec;
} options_t;

void usage(char *progname, int opt);

int main(int argc, char** argv)
{

    int p; /* number of processes */
    int my_rank; /* rank of process */
    int source; /* rank of sender */
    int dest; /* rank of receiver */
    MPI_Status status; /* status for receive */

    int opt;
    /* initial value of program parameter*/
    options_t options = {0, SEC};
    opterr = 0;

    while ((opt = getopt(argc, argv, OPTSTR)) != EOF) 
        switch(opt) {
            case 's':
                options.sec = atoi(optarg);
                break;
            case 'v':
                options.verbose += 1;
                break;
            case 'h':
            default:
                usage(basename(argv[0]), opt);
                /* NOTREACHED */
                break;
        }    


    double time1, time2, time_diff= 0.0; /*timer for each program*/
    int count =0;

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
    time1=MPI_Wtime();
    //printf("time send %lf",time1);
    do{
        /* initialize the random value of x and y */
        x = ((float)rand()/(float)(RAND_MAX)) * SCALE;
        y = ((float)rand()/(float)(RAND_MAX)) * SCALE;

        result = Eggholder(x,y);
        if(result<local_min){
            local_min = result;
        }
        count++;
        if(count>N){
            time2=MPI_Wtime();
            time_diff = time2 -time1;
            //printf("time difference =  %lf\n",time_diff);
            count = 0;
        }
    }while(time_diff<options.sec);

    printf("the local minimun found within %fsecs of process(%d) is %f\n",time_diff, my_rank, local_min );

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
    return EXIT_SUCCESS;
}

void usage(char *progname, int opt) {
   fprintf(stderr, USAGE_FMT, progname?progname:DEFAULT_PROGNAME);
   exit(EXIT_FAILURE);
   /* NOTREACHED */
}
