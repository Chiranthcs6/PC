/* Program 7: MPI_Bcast - root sends the same value to every process
 * Compile: mpicc p7_broadcast.c -o p7
 * Run:     mpirun -np 4 ./p7
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, number = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        number = 42;
        printf("Process 0 is broadcasting number %d\n", number);
    }

    /* (data, count, type, root, communicator) - called by ALL processes */
    MPI_Bcast(&number, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received number %d\n", rank, number);

    MPI_Finalize();
    return 0;
}
