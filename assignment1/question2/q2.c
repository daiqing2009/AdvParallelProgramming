/* main_parallel_matrix_vector.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>//to include memcpy
#include <stdbool.h>
#include "mpi.h"

#define FILE_NAME_LEN   50
#define PROMPT_LEN      80
#define SHOW_DIG_LEN    3

void Init_local_matrix(float *local_A, int m, int n, int my_rank, int p);

void Gather_save_matrix(int m, int n, int my_rank, int p);

void Read_scatter_matrix(int m, int n, int my_rank, int pi);

int main(int argc, char* argv[]) 
{
    int             my_rank, p;
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

    double time1, time2, time3;

    //add profiling timer
    if(0==my_rank){
        time1 = MPI_Wtime();
    }

    Gather_save_matrix(m, n, my_rank, p);

    MPI_Barrier(MPI_COMM_WORLD);
    //add profiling timer
    if(0==my_rank){
        time2 = MPI_Wtime();
    }   
    Read_scatter_matrix(m, n, my_rank, p);
    // add profiling timer
    MPI_Barrier(MPI_COMM_WORLD);
    if(0==my_rank){
        time3 = MPI_Wtime();
        //Performance summary
        printf("The time elapsed for Gather_save_matrix:%.2f s\n", time2-time1);
        printf("The time elapsed for Read_scatter_matrix:%.2f s\n", time3-time2);
    }
    MPI_Finalize();

    return EXIT_SUCCESS;
}  

void Print_matrix(char *prompt,float *mat, int m, int n){
    int i, j;
    printf("%s\n", prompt);
    // only print left-most portion of matrix when too large
    bool print_line=false;
    for(int i = 0; i < m; i++) {
        
        for(int j = 0; j < n; j++) {
            if((i+j)<SHOW_DIG_LEN||(i+j)>n+m-2-SHOW_DIG_LEN){
                print_line = true;
                printf("%.2f ", mat[i*n+j]);
            }
            if((i+j)==SHOW_DIG_LEN&&((i+j)<n+m-2-SHOW_DIG_LEN)){
                print_line = true;
                printf(" ... ");
            }
        }
        if(print_line)
            printf("\n");
    } 
}

void Init_local_matrix(float *local_A ,int m, int n,int my_rank, int p)
{
    char prompt [PROMPT_LEN];
    snprintf(prompt,PROMPT_LEN , "Printing the initial matrix of process(%d)",my_rank);

    int local_m = m/p;
    int local_n = n/p;

    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < local_n; j++)
            local_A[i*local_n+j] = (float)(i*n+my_rank*local_n+j);

    //Print_matrix(prompt, local_A, m, local_n);
}

/** Fulfill the requieremnt of section a: 
 * process 0 recieved matrix from other proceses and save in sequence
 * While recieving the matrix, proceess 0 print them immediately
 */
void Gather_save_matrix(int m, int n, int my_rank, int p)
{
    FILE *fp;
    char file_name[FILE_NAME_LEN];

    char prompt [PROMPT_LEN];
    int local_m = m/p;
    int local_n = n/p;
    float           *local_A; 
    float           *global_row;
    local_A=(float *)malloc(m*local_n*sizeof(float));
    global_row=(float *)malloc(local_m*n*sizeof(float));

    Init_local_matrix(local_A,m,n,my_rank,p);

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

           }
            snprintf(file_name,FILE_NAME_LEN , "matrix_p%d.data", i);
            if(NULL==(fp = fopen(file_name,"wb+"))){
                printf("Cant initializd matrix.data");
                fclose(fp);
                exit(EXIT_FAILURE);
            }
            
            snprintf(prompt, PROMPT_LEN, "After Recieved matrix from process(%d) batch(%d)",j,i);
            Print_matrix(prompt,&(global_row[0]),local_m,n);

            // write to file once global_row is filled
            fwrite(global_row,sizeof(float),local_m*n,fp);
            fclose(fp);
            printf("finish writing to file by process(0) for batch(%d)\n",i ); 
        }
    }
    else{
        int i, index_send;
        for(i=0;i<p;i++){
            snprintf(prompt, PROMPT_LEN, "Sending  matrix from process(%d) batch(%d)",my_rank,i);
            index_send = i*local_m*local_n;
            //Print_matrix(prompt,&(local_A[index_send]), local_m, local_n);
            MPI_Send(&(local_A[index_send]),local_m*local_n,MPI_FLOAT,0,i,MPI_COMM_WORLD);
        }

    }

    free(local_A);
    free(global_row);

} 

/** Fulfill the requieremnt of section b: 
 * process 0 read row by row of the matrix saved in file
 * and scat ter n/p portion to other processes
 */
void Read_scatter_matrix(int m, int n, int my_rank, int p){
    FILE *fp;
    char file_name[FILE_NAME_LEN];

    char prompt [PROMPT_LEN];

    int local_m = m/p;
    int local_n = n/p;

    float           *local_A; 
    float           *global_row;
    local_A=(float *)malloc(m*local_n*sizeof(float));
    global_row=(float *)malloc(local_m*n*sizeof(float));
    // Print_matrix("Emptied global_row",&(global_row[0]),local_m,n);

    MPI_Status status;
    MPI_Datatype column_mpi_t;
    MPI_Type_vector(local_m*local_n,local_n,n,MPI_FLOAT, &column_mpi_t);
    MPI_Type_commit(&column_mpi_t);
    
    if (my_rank==0){
        int i,j,index_send;
        int ig,jg,ret;
        i = 0;
        for(i=0;i<p;i++){
            snprintf(file_name,FILE_NAME_LEN , "matrix_p%d.data", i);
            if(NULL== (fp = fopen(file_name,"rb"))){
                printf("Cant read matrix.data");
                fclose(fp);
                exit(EXIT_FAILURE);
            }

            // read from file 
            fread(global_row, sizeof(float), local_m*n, fp);

            snprintf(prompt,PROMPT_LEN , "Read  matrix of batch(%d)",i);
            Print_matrix(prompt,&(global_row[0]),local_m,n);

            for(j=1;j<p;j++){
                index_send=j*local_n;
                MPI_Send(&(global_row[index_send]),1,column_mpi_t,j,i,MPI_COMM_WORLD);          
            }
            printf("process 0 has finished sending batch(%d)\n",i);
        }
        //TODO: nice to have ~ copy data to local_A of process 0;
        printf("process 0 has finished reading the file\n");
        fclose(fp);
    }
    else{
        int i, index_rec;
        for(i=0;i<p;i++){
            index_rec = i*local_m*local_n;
            MPI_Recv(&(local_A[index_rec]),local_m*n,MPI_FLOAT,0,i,MPI_COMM_WORLD,&status);
            //MPI_Recv(&(global_row[local_n/2]),1,column_mpi_t,0,i,MPI_COMM_WORLD,&status);
            //MPI_Recv(&(global_row[local_n/2]),local_m*local_n,MPI_FLOAT,0,i,MPI_COMM_WORLD,&status);

            //Print_matrix(prompt,&(global_row[0]),local_m,n);

        }
        
        snprintf(prompt, PROMPT_LEN, "Recieved matrix at process(%d) after (%d) batches",my_rank, i);
        Print_matrix(prompt,&(local_A[0]), m, local_n);
    }
    free(local_A);
    free(global_row);
}

