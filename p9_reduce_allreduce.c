/* Program 9: MPI_Reduce and MPI_Allreduce with MAX, MIN, SUM, PROD
 * MPI_Reduce    -> result only on root (process 0)
 * MPI_Allreduce -> result on every process
 * Compile: mpicc p9_reduce_allreduce.c -o p9
 * Run:     mpirun -np 4 ./p9
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, value;
    int sum, prod, max, min;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    value = rank + 1;                /* values: 1, 2, 3, ... */
    printf("Process %d has value %d\n", rank, value);
    fflush(stdout);

    /* ---------- MPI_Reduce (answer only at root) ---------- */
    MPI_Reduce(&value, &sum,  1, MPI_INT, MPI_SUM,  0, MPI_COMM_WORLD);
    MPI_Reduce(&value, &prod, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);
    MPI_Reduce(&value, &max,  1, MPI_INT, MPI_MAX,  0, MPI_COMM_WORLD);
    MPI_Reduce(&value, &min,  1, MPI_INT, MPI_MIN,  0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nMPI_Reduce (root only): Sum=%d Prod=%d Max=%d Min=%d\n\n",
               sum, prod, max, min);
        fflush(stdout);
    }

    /* ---------- MPI_Allreduce (answer at every process) ---------- */
    MPI_Allreduce(&value, &sum,  1, MPI_INT, MPI_SUM,  MPI_COMM_WORLD);
    MPI_Allreduce(&value, &prod, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &max,  1, MPI_INT, MPI_MAX,  MPI_COMM_WORLD);
    MPI_Allreduce(&value, &min,  1, MPI_INT, MPI_MIN,  MPI_COMM_WORLD);

    printf("Process %d Allreduce: Sum=%d Prod=%d Max=%d Min=%d\n",
           rank, sum, prod, max, min);

    MPI_Finalize();
    return 0;
}
