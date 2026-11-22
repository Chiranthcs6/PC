/* Program 8: MPI_Scatter and MPI_Gather
 * Root splits an array (one value per process), each process changes its
 * value, then root collects all values back.
 * Compile: mpicc p8_scatter_gather.c -o p8
 * Run:     mpirun -np 4 ./p8
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, value;
    int data[100], result[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Process 0 data: ");
        for (int i = 0; i < size; i++) {
            data[i] = i * 10;
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    /* Scatter: process i gets data[i] */
    MPI_Scatter(data, 1, MPI_INT, &value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d received %d\n", rank, value);

    value = value + rank;            /* each process does some work */

    /* Gather: result[i] = value from process i (only on root) */
    MPI_Gather(&value, 1, MPI_INT, result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Process 0 gathered: ");
        for (int i = 0; i < size; i++)
            printf("%d ", result[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
