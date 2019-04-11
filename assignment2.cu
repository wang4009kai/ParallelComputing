#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

// A small prime number used for modulus operation.
const int p = 103;

void generateRandomArray(int *rndArray,unsigned long long int n);
void resultVerification(int *A, int *B, int *prod,unsigned long long int n);
void initArray(int *A,unsigned long long int n);
unsigned long long int ipow(unsigned long long int base, int exp);
bool verify(int *array1, int *array2,unsigned long long int n);
void polynomialMultiplication(int *A, int *B, int *result, unsigned long long int n, int t);
void printArray(int *A,unsigned long long int n);

// Cuda kernal for multiplying two polynomial and store the result in the result array.
__global__ void PolyMulCUDA(int *A, int *B, int *result, unsigned long long int n) {
    unsigned long long int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int i = id % n;
    unsigned long long int j = id / n;
    unsigned long long int index;

    if ( (i + j) < n) {
    	index  = (i + j) * n + j;	
    } else {
    	index = (i + j) * n + n - 1 - i;
    }
    result[index] = (A[i] * B[j]) % p;
    //printf("position is %llu, value is %d \n", index, result[index]); 
}

__global__ void ParallelReductionCUDA(int *result, unsigned long long int n) {
    unsigned long long int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Start parallel reduction.
    for (long long int i=1; i<n; i++) {
        // Use a total of 2n -1 processors for reduction.
        if (id < (2*n -1)) {
            // Sum up all n terms for each degree of x.
            result[id * n] = (result[id * n] + result[id * n + i]) % p;
            //printf("index is %llu, value is %d \n", id*n, result[id*n]);
            // Clean up the used field.
            result[id * n + i] = 0;
        }
    }
}

__global__ void CombineCUDA(int *result, unsigned long long int n) {
    unsigned long long int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Start parallel reduction.
    // Remove the gap in the result array after the parallel reduction.
    // Use a total of 2n -1 processors for gap removal.
    if (id < (2*n -1)) {
    // There is a n-1 gap between each value, remove them.
            result[id] = result[n * id];
    }  
}

// Generate a random array used as the polynomial.
void generateRandomArray(int *rndArray,unsigned long long int n) {
    for (unsigned long long int i = 0; i < n; i++){
        rndArray[i] = rand() % p;
    }
}

void resultVerification(int *A, int *B, int *prod,unsigned long long int n)
{
    // Initialize the porduct polynomial
    initArray(prod, 2*n-1);
    
    // Multiply two polynomials term by term
    // Take ever term of first polynomial
    for (unsigned long long int i=0; i<n; i++)
    {
        // Multiply the current term of first polynomial
        // with every term of second polynomial.
        for ( unsigned long long int j=0; j<n; j++) {
            prod[i+j] = (prod[i+j] + A[i]*B[j]) % p;
        }
    }
}

void initArray(int *A,unsigned long long int n) {
    // Initialize the porduct polynomial
    for (unsigned long long int i = 0; i<n; i++)
        A[i] = 0;
}

unsigned long long int ipow(unsigned long long int base, int exp) {
    unsigned  long long int result = 1;
    while(exp) {
    	if (exp & 1){
	  result *= base;
	}
	exp >>=1;
	base *= base;
    }
    return result;
}

bool verify(int *array1, int *array2,unsigned long long int n) {
   for (unsigned long long int i=0; i<(2*n -1); i++) {
 	if (array1[i] != array2[i]) {
		printf("index: %d is different \n", i);
		return false;
	}
   }
   return true;
}

void printArray(int *A,unsigned long long int n){
    for (unsigned long long int i=0; i<n; i++) {
        printf("array value is: %d \n", A[i]);
    }
}

void polynomialMultiplication(int *A, int *B, int *result, unsigned long long int n, int t) {

    // Setup execution parameters
    dim3 thread_per_block(t);
    dim3 numOfBlock(n * n / t);

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
    ParallelReductionCUDA <<<numOfBlock, thread_per_block >>>(dev_z, n);
    CombineCUDA <<<numOfBlock, thread_per_block >>>(dev_z, n);

    cudaMemcpy( result, dev_z, (2*n - 1) * sizeof(int), cudaMemcpyDefault );

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
}

int main() {
    int degree;
    // Get the user to input the size of two polynomials.
    printf("Please enter the degree of the polynomial (base 2): ");
    scanf("%d", &degree);
    printf("You entered: %d \n", degree);
    
    unsigned long long int n = ipow(2,degree);

    // Set the rand() using the time as seed.
    time_t t;
    srand((unsigned) time(&t));

    int arrayA[n];
    int arrayB[n];
    int prod[2*n -1];
    int result[2*n -1];

    // Generate two random array represeting two polynomials.
    generateRandomArray(arrayA, n);
    
    generateRandomArray(arrayB, n);

    //printArray(arrayA, n);
    //printArray(arrayB, n);
    
    resultVerification(arrayA, arrayB, prod, n);

    int config;
    // Get the user to input whether the program will use n threadblock or n^2/t threadblock.
    printf("Please enter the config 1 for n thread block and 2 for n^2/t thread block: ");
    scanf("%d", &config);

    if (config == 1) {
   	polynomialMultiplication(arrayA, arrayB, result, n, 1);
    } else if (config == 2) {
    	int t;
   	// Get the user to input value t.
   	printf("Please enter the number of thread per block: ");
   	scanf("%d", &t);	
	polynomialMultiplication(arrayA, arrayB, result, n, t);
    }

    //printArray(prod, 2*n -1);
    //printf("second array \n");
    //printArray(result, 2*n-1);

    if (verify(prod, result, n)) printf("the output is correct \n");
    else printf("incorrect output \n");

    return 0;
}
