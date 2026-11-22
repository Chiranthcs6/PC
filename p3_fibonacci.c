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
