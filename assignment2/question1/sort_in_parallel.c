/*
 * Sort random number in parallel
 * 1. Genereate random number in each processs independently
 * 2. Communicate to related process according to the value of numbers
 * 3. Sort within each process
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define PRE_POST_INT 5

void merge(float arr1[], float arr2[], int n1, int n2, float arr3[]);

int cmpfunc(const void *a, const void *b);

void print_array(char *prompt, float arr[], int len);

int main(int argc, char *argv[])
{
    int p, my_rank, N, i, j, tag1 = 0, tag2 = 0;
    float *num_array, *sorted_array, *cat_array;
    float *send_buf, *recv_buf;
    int len_cat, len_recv;
    char prompt[70];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Request reqs[p - 1];
    // MPI_Status stats[p - 1];

    /* Read the input of N */
    if (my_rank == 0)
    {
        printf("Please enter the number of random array length N: \n");
        scanf("%d", &N);
        // TODO: make sure N must be bigger than Zero
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* genarate Random Number array */
    // printf("Initializing the array of len(%d) for proc(%d) \n", N, my_rank);

    num_array = malloc(N * sizeof(float));
    srand(my_rank); // make sure every random number have different seed
    for (i = 0; i < N; i++)
    {
        num_array[i] = (float)rand() / (float)(RAND_MAX);
    }
    // sprintf(prompt, "The initialized array of process(%d) ", my_rank);
    // print_array(prompt, num_array, N);

    /* sort on the generated array*/
    qsort(num_array, N, sizeof(float), cmpfunc);

    sprintf(prompt, "Locally sorted initialized array(len=%d) of process(%d)  ", N, my_rank);
    print_array(prompt, num_array, N);

    /* initial sendbuf for each process */
    send_buf = malloc(p * N * sizeof(float));
    int pre_bucket, cur_bucket = 0;
    int offset = 0;
    for (i = 0; i < N; i++)
    {
        cur_bucket = (int)floorf(p * num_array[i]);

        while (pre_bucket < cur_bucket)
        {
            while (offset < N)
            {
                send_buf[pre_bucket * N + offset] = EOF;
                offset++;
            }
            offset = 0;
            pre_bucket ++;
        }
        send_buf[cur_bucket * N + offset] = num_array[i];
        offset++;
    }
    /* fill remaining bucket*/
    do
    {
        while (offset < N)
        {
            send_buf[cur_bucket * N + offset] = EOF;
            offset++;
        }
        cur_bucket++;
        offset = 0;
    } while (cur_bucket <= p);

    /* dispatch number to corresponding process*/
    sprintf(prompt, "sending arrayes to correspinding proceses from process(%d)\n", my_rank);
    print_array(prompt, send_buf, N * p);

    for (j = 0; j < p; j++)
    {
        if (j != my_rank)
        {
            MPI_Isend(&send_buf[j * N], N, MPI_FLOAT, j, tag1,
                      MPI_COMM_WORLD, &reqs[j]);
        }
    }

    /* sort on recieved batch of number asynchronizedly */
    recv_buf = malloc(p * N * sizeof(float));
    // printf("recieving arrayes from correspinding proceses to process(%d)\n", my_rank);
    for (j = 0; j < p; j++)
    {
        MPI_Irecv(&recv_buf[j * N], N, MPI_FLOAT, j, tag2,
                  MPI_COMM_WORLD, &reqs[j]);
    }

    /* merge recieved array(sorted) with existing cancacated array */
    int proc_recv = -1;
    sorted_array = malloc(p * N * sizeof(float));
    cat_array = malloc(p * N * sizeof(float));
    /* copy the remaining part that don't need to send*/
    memcpy(sorted_array, &send_buf[my_rank * N], N * sizeof(float));

    sprintf(prompt, "The initialzied sorted array of process(%d): ", my_rank);
    print_array(prompt, sorted_array, p* N);

    for (j = 1; j < p; j++)
    {
        MPI_Waitany(p, reqs, &proc_recv, MPI_STATUS_IGNORE);
        // sprintf(prompt, "recieved one response form proc(%d)", proc_recv);
        // print_array(prompt, &recv_buf[proc_recv * N], N);
        memcpy(cat_array, sorted_array, j * N * sizeof(float));
        merge(cat_array, &recv_buf[proc_recv * N], j * N, N, sorted_array);
    }
    // sprintf(prompt, "The sorted array of process(%d) before purge: ", my_rank);
    // print_array(prompt, sorted_array, p* N);

    /* remove all dummy placer and print the end result*/
    int len_sroted = 0;
    /* purge EOFs in the array */
    memcpy(cat_array, sorted_array, j * N * sizeof(float));
    for (i = 0; i < p * N; i++)
    {
        if (cat_array[i] > EOF)
        {
            sorted_array[len_sroted] = cat_array[i];
            len_sroted++;
        }
    }

    sprintf(prompt, "The sorted array of process(%d): ", my_rank);
    print_array(prompt, sorted_array, len_sroted);

    free(sorted_array);
    free(cat_array);
    free(num_array);
    free(recv_buf);
    free(send_buf);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

/**
 * compare the integer array
 */
int cmpfunc(const void *a, const void *b)
{
    if (*(float *)a < *(float *)b)
        return -1;
    if (*(float *)a == *(float *)b)
        return 0;
    if (*(float *)a > *(float *)b)
        return 1;
    return 0;
}

/**
 * Merge two already sorted array
 */
void merge(float arr1[], float arr2[], int n1, int n2, float arr3[])
{
    int i = 0, j = 0, k = 0;

    while (i < n1 && j < n2)
    {
        if (arr1[i] < arr2[j])
            arr3[k++] = arr1[i++];
        else
            arr3[k++] = arr2[j++];
    }

    while (i < n1)
        arr3[k++] = arr1[i++];

    while (j < n2)
        arr3[k++] = arr2[j++];
}

/**
 * print integer array in a human friendly way
 */
void print_array(char *prompt, float arr[], int len)
{
    char result[len * 8];
    char num_ele[8];
    int i = 0;
    for (; i < len; i++)
    {
        sprintf(num_ele, "%.3f,", arr[i]);
        strcat(result, num_ele);
    }
    printf("%s: %s \n", prompt, result);
}