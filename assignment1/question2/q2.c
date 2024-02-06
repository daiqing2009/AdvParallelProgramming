/* main_parallel_matrix_vector.c */
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
//#include "matrix_vector.h"

void Init_local_matrix(float local_A[],
        int m, int n, int my_rank, int p);

void Gather_save_matrix(float local_A[],
        int m, int n, int my_rank, int p, float gloabal_row[]);

void Read_scatter_matrix();

int main(int argc, char* argv[]) 
{
    int             my_rank, p;
    float           *local_A; 
    float           *global_row;
    int             m, n;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) 
    {
        printf("Enter the dimensions of the matrix (m x n)\n");
        scanf("%d %d", &m, &n);
        //TODO: check if m is equal to n for a symmatric matrix
        //TODO: make sure m & n are both evenly divided by p
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_m = m/p;
    int local_n = n/p;
    local_A=(float *)malloc(m*local_n*sizeof(float));
    global_row=(float *)malloc(local_m*n*sizeof(float));

    /* Init local matrix of each process before gather at process 0 */
    Init_local_matrix(local_A, m, n, my_rank, p);
    //TODO: add profiling timer
    Gather_save_matrix(local_A, m, n, my_rank, p,global_row);
    //TODO: add profiling timer
    Read_scatter_matrix();
    //TODO: add profiling timer

    //TODO: print the performance

    MPI_Finalize();

    free(local_A);
    free(global_row);

    return 0;
}  

void Print_matrix(char *prompt,float *mat, int m, int n){
    int i, j;
    printf("%s\n", prompt);
    //TODO: only print left-most portion of matrix when too large
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%.2f ", mat[i*n+j]);
        }
        printf("\n");
    } 
}

void Init_local_matrix(float local_A[], int m, int n,
        int my_rank, int p)
{
    char prompt [100];
    snprintf(prompt, 100, "Printing the initial matrix of process(%d)",my_rank);

    int local_m = m/p;
    int local_n = n/p;

    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < local_n; j++)
            local_A[i*local_n+j] = (float)(i*n+my_rank*local_n+j);

    Print_matrix(prompt, local_A, m, local_n);
}

/** Fulfill the requieremnt of section a: 
 * process 0 recieved matrix from other proceses and save in sequence
 * While recieving the matrix, proceess 0 print them immediately
 */
void Gather_save_matrix(float local_A[], int m, int n, 
        int my_rank, int p, float global_row[])
{
    char prompt [100];

    int local_m = m/p;
    int local_n = n/p;
    /*
       MPI_Gather(local_A, local_m*n, MPI_FLOAT, 
       temp_matrix, local_m*n, MPI_FLOAT, 
       0, MPI_COMM_WORLD);
       */
    MPI_Status status;
    MPI_Datatype column_mpi_t;
    MPI_Type_vector(local_m*local_n,local_n,n,MPI_FLOAT, &column_mpi_t);
    MPI_Type_commit(&column_mpi_t);

    if (my_rank==0){
        int i,j;
        for(i=0;i<p;i++){
            for(j=0;j<p;j++){
                MPI_Recv(&(global_row[j*local_n]),1,column_mpi_t,j,i,MPI_COMM_WORLD,&status);

                snprintf(prompt, 100, "Recieving  matrix from process(%d)",my_rank);
                Print_matrix(prompt,&(global_row[i*local_m*n + j*local_n]),local_m,local_n);

            }
            //TODO: write to file once global_row is filled
            break; 
        }
    }
    else{
        int i;
        for(i=0;i<p;i++){
            MPI_Send(&(local_A[i*local_m]),local_m*local_n,MPI_FLOAT,0,i,MPI_COMM_WORLD);
        }

    }

} 

/** Fulfill the requieremnt of section b: 
 * process 0 read row by row of the matrix saved in file
 * and scat ter n/p portion to other processes
 */
void Read_scatter_matrix(){


}

