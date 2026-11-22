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
