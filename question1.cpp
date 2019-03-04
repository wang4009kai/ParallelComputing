#include <iostream>
#include <assert.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilktools/cilkview.h>
#include <math.h>
#include <thread>
#include <ctime>


using namespace std;

void sumArray(vector<int> *A, vector<int> *B, vector<int> *D, int n) {
    int grain_size = sqrt(n);
    int r = (n-1)/grain_size +1;
    thread threadCollection[r];
    for(int i=0; i<r; i++) {
        thread t1(addSubarray(A, B, D, r*grain_size, min((i+1)*grain_size, n)));
        threadCollection[i] = t1;
    }
    
    for(int i=0; i<r; i++){
        threadCollection[i].join();
    }
}

void addSubarray(vector<int> *A, vector<int> *B, vector<int> *D, int i, int j) {
    for(int k=i; i<j; k++) {
        (*D)[k] = (*A)[k] + (*B)[K];
    }
}

int main(int argc, char * argv[]){
    
    if (argc == 1)
    {
        cout<<"Usage: ./workers order [check]"<<endl;
        exit(0);
    }
    else if ( argc > 2 )
    {
        __cilkrts_set_param("nworkers", argv[1]);
    }
    
    int n=4096;
    vector<int> A(n);
    vector<int> B(n);
    vector<int> D(n);
    
    srand((unsigned)time(0));
    
    for(int i=0; i<size; i++){
        A[i] = (rand()%100)+1;
        B[i] = (rand()%100)+1;
    }
    
    double start = __cilkview_getticks();
    sumArray(&A, &B, &D, n);
    double end = __cilkview_getticks();
    double run_time = (end - start) / 1.f;
    
}




