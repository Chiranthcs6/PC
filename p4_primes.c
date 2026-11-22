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
