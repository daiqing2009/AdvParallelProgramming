/* main_parallel_matrix_vector.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>//to include memcpy
#include "mpi.h"

#define MATRIX_FILE_NAME    "matrix.data"

void Init_local_matrix(float local_A[],
        int m, int n, int my_rank, int p);

void Gather_save_matrix(float local_A[],
        int m, int n, int my_rank, int p, float global_row[]);

void Read_scatter_matrix(float local_A[],
        int m, int n, int my_rank, int p, float global_row[]);

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
    //TODO: empty lcoal_A and global_row to make sure the content was read from file
    //TODO: add profiling timer
    Read_scatter_matrix(local_A, m, n, my_rank, p,global_row);
    //TODO: add profiling timer

    //TODO: print the performance

    MPI_Finalize();

    free(local_A);
    free(global_row);

    return EXIT_SUCCESS;
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
    FILE *fp;

    //TODO: remove file in the directory if already exits

    if((fp = fopen(MATRIX_FILE_NAME,"a+"))==NULL){
        printf("Cant initializd matrix.data");
        exit(EXIT_FAILURE);
    }


    char prompt [100];

    int local_m = m/p;
    int local_n = n/p;

    MPI_Status status;
    MPI_Datatype column_mpi_t;
    MPI_Type_vector(local_m*local_n,local_n,n,MPI_FLOAT, &column_mpi_t);
    MPI_Type_commit(&column_mpi_t);

    if (my_rank==0){
        int i,j;        //index of block for copy
        int index_rec;  //index of starting index of block copy
        int i0,j0;      //index of process 0 copy
        for(i=0;i<p;i++){
            // copy the content of process 0 into global_row
            for(i0=0;i0<local_m;i0++){
                for(j0=0;j0<local_n;j0++){
                    global_row[i0*n+j0] = local_A[i*local_m*local_n+i0*local_n+j0];
                }   
            }
            // Print_matrix("global row after proc(0) copy", &(global_row[0]), local_m, n);
            for(j=1;j<p;j++){
                index_rec=j*local_n;
                MPI_Recv(&(global_row[index_rec]),1,column_mpi_t,j,i,MPI_COMM_WORLD,&status);

                snprintf(prompt, 100, "After Recieved matrix from process(%d) batch(%d)",j,i);
                Print_matrix(prompt,&(global_row[0]),local_m,n);

            }
            // write to file once global_row is filled
            fwrite(global_row,sizeof(float),local_m*n,fp);
            printf("finish writing to file by process(0) for batch(%d)\n",i ); 
        }
    }
    else{
        int i, index_send;
        for(i=0;i<p;i++){
            snprintf(prompt, 100, "Sending  matrix from process(%d) batch(%d)",my_rank,i);
            index_send = i*local_m*local_n;
            Print_matrix(prompt,&(local_A[index_send]), local_m, local_n);
            MPI_Send(&(local_A[index_send]),local_m*local_n,MPI_FLOAT,0,i,MPI_COMM_WORLD);
        }

    }
    fclose(fp);

} 

/** Fulfill the requieremnt of section b: 
 * process 0 read row by row of the matrix saved in file
 * and scat ter n/p portion to other processes
 */
void Read_scatter_matrix(float local_A[],
        int m, int n, int my_rank, int p, float global_row[]){
    FILE *fp;
    if((fp = fopen(MATRIX_FILE_NAME,"r"))==NULL){
        printf("Cant read matrix.data");
        exit(EXIT_FAILURE);
    }

    char prompt [100];

    int local_m = m/p;
    int local_n = n/p;

    MPI_Status status;
    MPI_Datatype column_mpi_t;
    MPI_Type_vector(local_m*local_n,local_n,n,MPI_FLOAT, &column_mpi_t);
    MPI_Type_commit(&column_mpi_t);

    if (my_rank==0){
        int i,j,index_send;
        int ig,jg,ret;
        i = 0;
        for(i=0;i<=p;i++){
            // read from file 
            for (ig = 0; ig < local_m; ++ig) {
                for (jg = 0; jg < n - 1; ++jg) {
                    if(EOF== fscanf(fp, "%f ", &global_row[ig*n+jg])){
                        printf("EOF reached!\n");
                        exit(EXIT_SUCCESS);
                    }
                }
                if(EOF == fscanf(fp, "%f", &global_row[ig*n - 1])){
                    printf("EOF reached!\n");
                    exit(EXIT_SUCCESS);
                }
                
            }
        
            for(j=1;j<p;j++){
                index_send=j*local_n;
                snprintf(prompt, 100, "Read and sendng matrix to process(%d) batch(%d)",j,i);
                Print_matrix(prompt,&(global_row[0]),local_m,n);

                MPI_Send(&(global_row[index_send]),1,column_mpi_t,j,i,MPI_COMM_WORLD);          
            }
            printf("process 0 has finished sending batch(%d)\n",i);
        }
        //TODO: nice to have ~ copy data to local_A of process 0;
        printf("process 0 has finished reading the file\n");

    }
    else{
        int i, index_rec;
        for(i=0;i<p;i++){
            index_rec = i*local_m*local_n;
            snprintf(prompt, 100, "Recieved matrix at process(%d) batch(%d)",my_rank, i);
            //MPI_Recv(&(local_A[index_rec]),local_m*local_n,MPI_FLOAT,0,i,MPI_COMM_WORLD,&status);
            //Print_matrix(prompt,&(local_A[index_rec]), local_m, local_n);
            MPI_Recv(&(global_row[local_n/2]),1,column_mpi_t,0,i,MPI_COMM_WORLD,&status);
            //MPI_Recv(&(global_row[local_n/2]),local_m*local_n,MPI_FLOAT,0,i,MPI_COMM_WORLD,&status);
 
            Print_matrix(prompt,&(global_row[0]),local_m,n);

        } 

    }
    fclose(fp);
}

