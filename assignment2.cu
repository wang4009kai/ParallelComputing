#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

// A small prime number used for modulus operation.
const int p = 103;

__global__ void PolyMulCUDA(int *A, int *B, int *result, long long int n) {
    long long int id = blockIdx.x * blockDim.x + threadIdx.x;
    long long int i = (id -1) % n;
    long long int j = (id -1) / n;
    long long int d = i + j;
    result[d * n + i + 1] = A[i + 1] * B[j + 1];

    for (long long int k = 1; k <= n; k = k * 2) {
        if (id % (2 * k)) == 0) {
            result[id] += result[id + k];
        }
    }

    if ((id -1) % n ==0 ) {
        result[(2 * n - 1) * n + id/n + 1] = result[id];
    }
}

void generateRandomArray(int *rndArray, long long int n) {
    for (long long int i = 0; i < n; i++) {
        rndArray[i] = rand() % p;
    }
}

void resultVerification(int *A, int *B, int *prod, long long int n)
{
    // Initialize the porduct polynomial
    initArray(prod, 2*n-1);
    
    // Multiply two polynomials term by term
    // Take ever term of first polynomial
    for (long long int i=0; i<n; i++)
    {
        // Multiply the current term of first polynomial
        // with every term of second polynomial.
        for (long long int j=0; j<n; j++) {
            prod[i+j] = (product[i+j] + A[i]*B[j]) % p;

            //printf("value at %d is: %d \n",(i+j), A[0]);
        }
    }
}

void initArray(int *A, long long int n) {
    // Initialize the porduct polynomial
    for (long long int i = 0; i<n; i++)
        A[i] = 0;
}

void polynomialMultiplication(int *A, int *B, int *result, int n) {
    // This will pick the best possible CUDA capable device, otherwise
    // override the device ID based on input provided at the command line
    int dev = findCudaDevice(argc, (const char **)argv);

    // Setup execution parameters
    dim3 thread_per_block(n);
    dim3 numOfBlock(n);

    // Allocate device memory for the result polynomial.
    int result[2*n-1];

    // Init result array with value 0.
    initArray(result, 2*n-1);

    // Init Cuda memory.
    int *dev_x, *dev_y, *dev_z;

    cudaMalloc( &dev_x, n * sizeof(int) );
    cudaMalloc( &dev_y, n * sizeof(int) );
    // 2n^2 -n is the max memory can be used before the parallel reduction.
    cudaMalloc( &dev_z, (n * (2*n - 1)) * sizeof(int) );

    // Copy from cpu to gpu.
    cudaMemcpy( dev_x, A, n * sizeof(int), cudaMemcpyDefault );
    cudaMemcpy( dev_y, B, n * sizeof(int), cudaMemcpyDefault );

    printf("Computing result using CUDA Kernel...\n");

    // Performs operation using CUDA kernel
    PolyMulCUDA <<< numOfBlock, thread_per_block >>>(dev_x, dev_y, dev_z, n);
}

int main() {
    // Get the user to input the size of two polynomials.
    long long int n = 10;
    //printf("Please input an integer value for the size of the polynomial: ");
    //scanf("%d", &n);
    //printf("You entered: %d\n", n);
    
    int arrayA[n];
    int arrayB[n];
    int prod[2*n -1];
    int result[2*n+1];

    // Generate two random array represeting two polynomials.
    generateRandomArray(arrayA, n);
    
    generateRandomArray(arrayB, n);
    
    for (int i=0; i<n; i++) {
        printf("arrayA value is: %d \n", arrayA[i]);
    }
    
    for (int i=0; i<n; i++) {
        printf("arrayB value is: %d \n", arrayB[i]);
    }
    
    resultVerification(arrayA, arrayB, prod, n);

    polynomialMultiplication(arrayA, ArrayB, result, n);
    
    return 0;
}
