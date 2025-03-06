#include<iostream>
#include<cstdlib>

using namespace std;


__global__ void vectorAddition(float *A, float *B, float *C, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        *(C+idx) = *(B+idx) + *(A+idx);
    }
}
void print(float *arr, int N){
    if(arr == nullptr){
        cout << "Error: cannot print a null array" << endl;
    }
    for(int i = 0;i < N;i++){
        cout << *(arr+i) << " ";
    }
    cout << endl;
}

void checkCudaError(cudaError_t err, const char *operation){
    if(err != cudaSuccess){
        cout << "CUDA Error in operation: " << operation << " --- Error: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

float *Initialize(int N){
    if(N <= 0){
        cout << "Invalid array size" << endl;
        return nullptr;
    }

    float *arr = new(nothrow) float[N];

    if(arr == nullptr){
        cout << "Array initialization error" << endl;
        return nullptr;
    }
    for(int i = 0; i < N; i++){
        *(arr + i) = static_cast<float>(i);
    }
    return arr;
}
int main(){

    const int N = 250;
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    float *A = nullptr,*B = nullptr,*C = nullptr, *d_a = nullptr , *d_b = nullptr , *d_c = nullptr;

    A = Initialize(N);
    if(A == nullptr){
        cout << "Failed to initialize array A" << endl;
        return 1;
    }
    B = Initialize(N);
    if(B == nullptr){
        delete [] A;
        A = nullptr;
        cout << "Failed to initialize array B" << endl;
        return 1;
    }
    C = Initialize(N);
    if(C == nullptr){
        delete [] A;
        delete [] B;
        A = nullptr;
        B = nullptr;
        cout << "Failed to initialize array C" << endl;
        return 1;
    }

    checkCudaError(cudaMalloc(&d_a,N * sizeof(float)) , "cudaMalloc d_a");
    checkCudaError(cudaMalloc(&d_b,N * sizeof(float)) , "cudaMalloc d_b");
    checkCudaError(cudaMalloc(&d_c,N * sizeof(float)) , "cudaMalloc d_c");

    checkCudaError(cudaMemcpy(d_a,A,N*sizeof(float),cudaMemcpyHostToDevice) , "cudaMemcpy d_a");
    checkCudaError(cudaMemcpy(d_b,B,N*sizeof(float),cudaMemcpyHostToDevice) , "cudaMemcpy d_b");

    vectorAddition<<<grid_size,block_size>>>(d_a,d_b,d_c,N);

    checkCudaError(cudaGetLastError() , "kernel launch");
    checkCudaError(cudaDeviceSynchronize() , "kernel execution");

    checkCudaError(cudaMemcpy(C,d_c,N*sizeof(float),cudaMemcpyDeviceToHost), "cudaMemcpy C");

    print(C,N);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete [] A;
    delete [] B;
    delete [] C;

    A = nullptr;
    B = nullptr;
    C = nullptr;

    return 0;

}
