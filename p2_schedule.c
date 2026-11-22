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
