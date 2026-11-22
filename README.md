# Parallel Computing Lab (BCS702) – All Programs

## Setup (Linux)
```bash
sudo apt install gcc libomp-dev openmpi-bin libopenmpi-dev
```

- **OpenMP** programs (1–4): compile with `gcc -fopenmp`, set threads with `export OMP_NUM_THREADS=4` (optional).
- **MPI** programs (5–9): compile with `mpicc`, run with `mpirun -np <processes>`.
- If `mpirun` says *not enough slots*, add `--oversubscribe`, e.g. `mpirun --oversubscribe -np 4 ./p7`.

## Quick command table

| # | File | Compile | Run |
|---|------|---------|-----|
| 1 | [p1_mergesort.c](#program-1) | `gcc -fopenmp p1_mergesort.c -o p1` | `./p1` |
| 2 | [p2_schedule.c](#program-2) | `gcc -fopenmp p2_schedule.c -o p2` | `./p2` |
| 3 | [p3_fibonacci.c](#program-3) | `gcc -fopenmp p3_fibonacci.c -o p3` | `./p3` |
| 4 | [p4_primes.c](#program-4) | `gcc -fopenmp p4_primes.c -o p4` | `./p4` |
| 5 | [p5_send_recv.c](#program-5) | `mpicc p5_send_recv.c -o p5` | `mpirun -np 2 ./p5` |
| 6a | [p6a_deadlock.c](#program-6a) | `mpicc p6a_deadlock.c -o p6a` | `mpirun -np 2 ./p6a` |
| 6b | [p6b_no_deadlock.c](#program-6b) | `mpicc p6b_no_deadlock.c -o p6b` | `mpirun -np 2 ./p6b` |
| 7 | [p7_broadcast.c](#program-7) | `mpicc p7_broadcast.c -o p7` | `mpirun -np 4 ./p7` |
| 8 | [p8_scatter_gather.c](#program-8) | `mpicc p8_scatter_gather.c -o p8` | `mpirun -np 4 ./p8` |
| 9 | [p9_reduce_allreduce.c](#program-9) | `mpicc p9_reduce_allreduce.c -o p9` | `mpirun -np 4 ./p9` |

---

## Program 1

**Sequential vs Parallel Merge Sort (OpenMP sections)** — file: `p1_mergesort.c`

```c
/* Program 1: Sequential vs Parallel Merge Sort using OpenMP sections
 * Compile: gcc -fopenmp p1_mergesort.c -o p1
 * Run:     ./p1
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Merge two sorted halves a[l..m] and a[m+1..r] */
void merge(int a[], int l, int m, int r) {
    int *temp = malloc((r - l + 1) * sizeof(int));
    int i = l, j = m + 1, k = 0;

    while (i <= m && j <= r)
        temp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i <= m) temp[k++] = a[i++];
    while (j <= r) temp[k++] = a[j++];

    for (i = l, k = 0; i <= r; i++, k++)
        a[i] = temp[k];
    free(temp);
}

/* Normal (sequential) merge sort */
void mergeSort(int a[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSort(a, l, m);
        mergeSort(a, m + 1, r);
        merge(a, l, m, r);
    }
}

/* Parallel merge sort: the two halves are sorted at the same time
 * by two different threads using OpenMP sections */
void parallelMergeSort(int a[], int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(a, l, m);        /* thread 1 sorts left half  */
            #pragma omp section
            mergeSort(a, m + 1, r);    /* thread 2 sorts right half */
        }
        merge(a, l, m, r);
    }
}

int main() {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);

    int *a1 = malloc(n * sizeof(int));
    int *a2 = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        a1[i] = rand() % 1000;
        a2[i] = a1[i];               /* same data for both sorts */
    }

    double start = omp_get_wtime();
    mergeSort(a1, 0, n - 1);
    double seqTime = omp_get_wtime() - start;

    start = omp_get_wtime();
    parallelMergeSort(a2, 0, n - 1);
    double parTime = omp_get_wtime() - start;

    if (n <= 20) {                   /* print only small arrays */
        printf("Sorted array: ");
        for (int i = 0; i < n; i++) printf("%d ", a2[i]);
        printf("\n");
    }
    printf("Sequential merge sort time: %f seconds\n", seqTime);
    printf("Parallel   merge sort time: %f seconds\n", parTime);
    printf("Difference                : %f seconds\n", seqTime - parTime);

    free(a1);
    free(a2);
    return 0;
}
```

**Compile & Run**

```bash
gcc -fopenmp p1_mergesort.c -o p1
./p1
```

**Sample Output**

```
Enter number of elements: 1000000
Sequential merge sort time: 0.126122 seconds
Parallel   merge sort time: 0.087695 seconds
Difference                : 0.038427 seconds
```

---

## Program 2

**Iteration distribution with schedule(static,2)** — file: `p2_schedule.c`

```c
/* Program 2: Show which thread runs which iterations with schedule(static,2)
 * Compile: gcc -fopenmp p2_schedule.c -o p2
 * Run:     OMP_NUM_THREADS=2 ./p2
 */
#include <stdio.h>
#include <omp.h>

int main() {
    int n;
    int owner[1000];                  /* owner[i] = thread that ran iteration i */

    printf("Enter number of iterations (max 1000): ");
    scanf("%d", &n);

    /* Iterations are given to threads in chunks of 2 */
    #pragma omp parallel for schedule(static, 2)
    for (int i = 0; i < n; i++)
        owner[i] = omp_get_thread_num();

    /* Print each chunk: consecutive iterations done by the same thread */
    int start = 0;
    for (int i = 1; i <= n; i++) {
        if (i == n || owner[i] != owner[start]) {
            printf("Thread %d : Iterations %d -- %d\n", owner[start], start, i - 1);
            start = i;
        }
    }
    return 0;
}
```

**Compile & Run**

```bash
gcc -fopenmp p2_schedule.c -o p2
export OMP_NUM_THREADS=2
./p2
```

**Sample Output**

```
Enter number of iterations (max 1000): 4
Thread 0 : Iterations 0 -- 1
Thread 1 : Iterations 2 -- 3
```

---

## Program 3

**Fibonacci numbers using OpenMP tasks** — file: `p3_fibonacci.c`

```c
/* Program 3: n Fibonacci numbers using OpenMP tasks
 * Compile: gcc -fopenmp p3_fibonacci.c -o p3
 * Run:     ./p3
 */
#include <stdio.h>
#include <omp.h>

int fib(int n) {
    int x, y;
    if (n < 2)
        return n;

    #pragma omp task shared(x)      /* task 1: compute fib(n-1) */
    x = fib(n - 1);

    #pragma omp task shared(y)      /* task 2: compute fib(n-2) */
    y = fib(n - 2);

    #pragma omp taskwait            /* wait for both tasks to finish */
    return x + y;
}

int main() {
    int n;
    printf("Enter number of Fibonacci terms: ");
    scanf("%d", &n);

    printf("Fibonacci series: ");
    #pragma omp parallel            /* create a team of threads */
    {
        #pragma omp single          /* only one thread creates the tasks */
        {
            for (int i = 0; i < n; i++)
                printf("%d ", fib(i));
        }
    }
    printf("\n");
    return 0;
}
```

**Compile & Run**

```bash
gcc -fopenmp p3_fibonacci.c -o p3
./p3
```

**Sample Output**

```
Enter number of Fibonacci terms: 10
Fibonacci series: 0 1 1 2 3 5 8 13 21 34
```

---

## Program 4

**Prime numbers 1..n with parallel for (serial vs parallel time)** — file: `p4_primes.c`

```c
/* Program 4: Prime numbers from 1 to n - serial vs parallel for
 * Compile: gcc -fopenmp p4_primes.c -o p4 -lm
 * Run:     ./p4
 */
#include <stdio.h>
#include <omp.h>

int isPrime(int num) {
    if (num < 2) return 0;
    for (int i = 2; i * i <= num; i++)   /* check divisors up to sqrt(num) */
        if (num % i == 0)
            return 0;
    return 1;
}

int main() {
    int n, count;
    printf("Enter the value of n: ");
    scanf("%d", &n);

    /* ---- Serial ---- */
    count = 0;
    double start = omp_get_wtime();
    for (int i = 1; i <= n; i++)
        if (isPrime(i)) count++;
    double serialTime = omp_get_wtime() - start;
    printf("Serial  : %d primes found\n", count);

    /* ---- Parallel ---- */
    count = 0;
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:count)
    for (int i = 1; i <= n; i++)
        if (isPrime(i)) count++;
    double parallelTime = omp_get_wtime() - start;
    printf("Parallel: %d primes found\n", count);

    if (n <= 100) {                     /* print the primes only for small n */
        printf("Primes: ");
        for (int i = 1; i <= n; i++)
            if (isPrime(i)) printf("%d ", i);
        printf("\n");
    }

    printf("Serial Time  : %f seconds\n", serialTime);
    printf("Parallel Time: %f seconds\n", parallelTime);
    return 0;
}
```

**Compile & Run**

```bash
gcc -fopenmp p4_primes.c -o p4
./p4
```

**Sample Output**

```
Enter the value of n: 2000000
Serial  : 148933 primes found
Parallel: 148933 primes found
Serial Time  : 0.293606 seconds
Parallel Time: 0.077882 seconds
```

---

## Program 5

**MPI_Send and MPI_Recv** — file: `p5_send_recv.c`

```c
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
```

**Compile & Run**

```bash
mpicc p5_send_recv.c -o p5
mpirun -np 2 ./p5
```

**Sample Output**

```
Process 0 sending number 100 to process 1
Process 1 received number 100 from process 0
```

---

## Program 6a

**Deadlock using point-to-point communication** — file: `p6a_deadlock.c`

```c
/* Program 6 (a): DEADLOCK using point-to-point communication
 * Both processes SEND first and then RECEIVE -> each waits for the other forever.
 *
 * MPI_Ssend (synchronous send) is used so the deadlock always happens.
 * (Plain MPI_Send of a small message is often buffered by MPI and may not hang.)
 *
 * Compile: mpicc p6a_deadlock.c -o p6a
 * Run:     mpirun -np 2 ./p6a        (program hangs -> press Ctrl+C)
 */
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, sendData, recvData;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        sendData = 100;
        printf("Process 0 sending... (waiting for process 1 to receive)\n");
        fflush(stdout);
        MPI_Ssend(&sendData, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);   /* stuck here */
        MPI_Recv(&recvData, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received %d\n", recvData);
    }
    else if (rank == 1) {
        sendData = 200;
        printf("Process 1 sending... (waiting for process 0 to receive)\n");
        fflush(stdout);
        MPI_Ssend(&sendData, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);   /* stuck here */
        MPI_Recv(&recvData, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 1 received %d\n", recvData);
    }

    MPI_Finalize();
    return 0;
}
```

**Compile & Run**

```bash
mpicc p6a_deadlock.c -o p6a
mpirun -np 2 ./p6a      # hangs (deadlock) -> press Ctrl+C
```

**Sample Output**

```
Process 0 sending... (waiting for process 1 to receive)
Process 1 sending... (waiting for process 0 to receive)
(program hangs here)
```

---

## Program 6b

**Avoiding deadlock by changing the call order** — file: `p6b_no_deadlock.c`

```c
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
```

**Compile & Run**

```bash
mpicc p6b_no_deadlock.c -o p6b
mpirun -np 2 ./p6b
```

**Sample Output**

```
Process 0 received 200
Process 1 received 100
```

---

## Program 7

**MPI_Bcast (Broadcast)** — file: `p7_broadcast.c`

```c
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
```

**Compile & Run**

```bash
mpicc p7_broadcast.c -o p7
mpirun -np 4 ./p7
```

**Sample Output**

```
Process 0 is broadcasting number 42
Process 0 received number 42
Process 2 received number 42
Process 1 received number 42
Process 3 received number 42
```

---

## Program 8

**MPI_Scatter and MPI_Gather** — file: `p8_scatter_gather.c`

```c
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
```

**Compile & Run**

```bash
mpicc p8_scatter_gather.c -o p8
mpirun -np 4 ./p8
```

**Sample Output**

```
Process 0 data: 0 10 20 30
Process 0 received 0
Process 1 received 10
Process 2 received 20
Process 3 received 30
Process 0 gathered: 0 11 22 33
```

---

## Program 9

**MPI_Reduce and MPI_Allreduce (MAX, MIN, SUM, PROD)** — file: `p9_reduce_allreduce.c`

```c
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
```

**Compile & Run**

```bash
mpicc p9_reduce_allreduce.c -o p9
mpirun -np 4 ./p9
```

**Sample Output**

```
Process 0 has value 1
Process 1 has value 2
Process 2 has value 3
Process 3 has value 4

MPI_Reduce (root only): Sum=10 Prod=24 Max=4 Min=1

Process 0 Allreduce: Sum=10 Prod=24 Max=4 Min=1
Process 1 Allreduce: Sum=10 Prod=24 Max=4 Min=1
Process 2 Allreduce: Sum=10 Prod=24 Max=4 Min=1
Process 3 Allreduce: Sum=10 Prod=24 Max=4 Min=1
```

---

**Note:** for small inputs the parallel version may be *slower* (thread start-up cost). Use large n (e.g. 1000000) to see the speed-up. MPI output order between processes may vary from run to run.
