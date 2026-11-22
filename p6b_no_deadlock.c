/* Program 6 (b): AVOIDING deadlock by changing the call order
 * Process 0: SEND then RECEIVE
 * Process 1: RECEIVE then SEND   <- order swapped, so no deadlock
 *
 * Compile: mpicc p6b_no_deadlock.c -o p6b
 * Run:     mpirun -np 2 ./p6b
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, sendData, recvData;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        sendData = 100;
        MPI_Ssend(&sendData, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);   /* send first    */
        MPI_Recv(&recvData, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received %d\n", recvData);
    }
    else if (rank == 1) {
        sendData = 200;
        MPI_Recv(&recvData, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); /* receive first */
        MPI_Ssend(&sendData, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("Process 1 received %d\n", recvData);
    }

    MPI_Finalize();
    return 0;
}
