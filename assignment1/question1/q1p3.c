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

#define N   300000 /*number counter to check time after N iterations*/
#define SEC 5 /* the nnumber of seconds processes should keep running */
#define DIFF 0.1 /* define the */
#define SCALE 512 /* the scale of random parameters */
/* define and prompt the the parameter of program */
#define OPTSTR "vi:s:d:h"
#define USAGE_FMT  "%s [-v] [-s seconds for differenc check interval] [-d difference ] [-h]"
#define DEFAULT_PROGNAME "q1p3"

typedef struct {
    int verbose;
    int sec;
    float diff;
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
    options_t options = {0, SEC, DIFF};
    opterr = 0;

    while ((opt = getopt(argc, argv, OPTSTR)) != EOF) 
        switch(opt) {
            case 's':
                options.sec = (int)atoi(optarg);
            case 'd':
                options.diff = (float)atof(optarg);
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


    double time1, time2, time2_last, time_diff= 0.0; /*timer for each program*/
    int count =0;

    int n =0; /* the trail of finding local_min of each process */
    float x, y = 0.0; /* the value of init */
    float result, local_min, global_min, last_global_min, global_min_diff = 1024.0; /* the init of minimums of eggholder function, maxium of possible function */

    /* Start up MPI */
    MPI_Init(&argc, &argv);
    /* Find out process rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    /* Find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* seed the random value */
    srand(my_rank);

    /* calculate the local minimum of each process */
    time2_last=time1=MPI_Wtime();
    //printf("time send %lf",time1);
    /* alwasy loop until non-significant improvement has been found*/
    while(1){
        /* initialize the random value of x and y */
        x = ((float)rand()/(float)(RAND_MAX)) * SCALE;
        y = ((float)rand()/(float)(RAND_MAX)) * SCALE;

        result = Eggholder(x,y);
        /* update local min if found smaller result*/
        if(result<local_min){
            local_min = result;
        }
        /* only check time after count tickes */
        count++;
        if(count>N){
            time2=MPI_Wtime();
            time_diff = time2 - time2_last;
            //printf("time difference =  %lf\n",time_diff);
            count = 0;
            /* observation time span is reach, check result difference*/
            if(time_diff > options.sec){
                time2_last = time2;/* reset tick*/
                /* find global_min of local_min calculated by each process */
                MPI_Allreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
                global_min_diff = last_global_min - global_min;
                last_global_min = global_min;
                /*once */
                if(global_min_diff<options.diff){
                    printf("local minimum found by process(%d) at the end is %f within(%.3f seconds)\n", my_rank, local_min, time2-time1 );
                    break;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(my_rank == 0)
    {
        /* compare diff between steps, if no significant, stop try*/
        printf("the global minimum found of eggholder function is %f with x =  %f and y = %f\n",
                global_min, x, y);
        printf("different found between checks is %f", global_min_diff);
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
