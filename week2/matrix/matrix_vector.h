/* matrix_vector.h */
#define MAX_ORDER 100

typedef float LOCAL_MATRIX_T[MAX_ORDER][MAX_ORDER];

void Read_matrix(char* prompt, LOCAL_MATRIX_T local_A, 
        int local_m, int n, int my_rank, int p);
void Read_vector(char* prompt, float local_x[], 
        int local_n, int my_rank, int p);
void Parallel_matrix_vector_prod(LOCAL_MATRIX_T local_A, 
        int m, 
        int n, float local_x[], 
        float global_x[], 
        float local_y[],
        int local_m, int local_n);
void Print_matrix(char* title, LOCAL_MATRIX_T local_A, 
        int local_m, int n, int my_rank, int p);
void Print_vector(char* title, float local_y[], 
        int local_m, int my_rank, int p);



