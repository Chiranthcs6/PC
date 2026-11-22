/* Program 5: MPI_Send and MPI_Recv
 * Compile: mpicc p5_send_recv.c -o p5
 * Run:     mpirun -np 2 ./p5
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, number;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* my process id   */
    MPI_Comm_size(MPI_COMM_WORLD, &size);   /* total processes */

    if (size < 2) {
        if (rank == 0) printf("Please run with at least 2 processes.\n");
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        number = 100;
        printf("Process 0 sending number %d to process 1\n", number);
        /* (data, count, type, destination, tag, communicator) */
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        /* (data, count, type, source, tag, communicator, status) */
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received number %d from process 0\n", number);
    }

    MPI_Finalize();
    return 0;
}
