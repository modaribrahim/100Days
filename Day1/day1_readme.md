# File: vectorAddition.cu

## Summary
Implemented a basic vector addition using CUDA. Designed a kernel to perform parallel addition of two arrays, with each thread computing the sum of a pair of values. Added error handling, memory management, and synchronization for robustness.

### Learned:
- Basics of writing and launching a CUDA kernel.
- Grid and block size configuration for parallel execution.
- Proper memory allocation and deallocation on both host (CPU) and device (GPU).
- Importance of CUDA error checking and synchronization.

### Reading:
- Read Chapter 1 of the PMPP book.
  - Learned about the fundamentals of parallel programming, CUDA architecture, and the GPU execution model.

---

## vectorAddition.cu - Detailed Explanation
This file documents the implementation of a basic vector addition program in CUDA, including error handling, memory management, and synchronization.

### Code Explanation

#### 1. Kernel Function - `vectorAddition`
```cpp
__global__ void vectorAddition(float *A, float *B, float *C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        *(C + idx) = *(B + idx) + *(A + idx);
    }
}
```
- **Purpose**: Each thread computes `C[idx] = A[idx] + B[idx]`.
- **Index Calculation**:  
  - `blockDim.x`: Number of threads per block.  
  - `blockIdx.x`: Index of the current block.  
  - `threadIdx.x`: Index of the current thread within the block.  
  - Final index: `idx = blockDim.x * blockIdx.x + threadIdx.x`.
- **Boundary Check**: `if (idx < N)` ensures threads don’t access memory outside the array bounds.

#### 2. Memory Allocation & Initialization
```cpp
checkCudaError(cudaMalloc(&d_a, N * sizeof(float)), "cudaMalloc d_a");
```
- **Device Memory Allocation**: Allocates memory on the GPU using `cudaMalloc`.
- **Size Calculation**: Uses `N * sizeof(float)` to allocate the correct number of bytes for `N` float elements.

```cpp
float *Initialize(int N) {
    float *arr = new(nothrow) float[N];  // Safe memory allocation
    if (arr == nullptr) {
        cout << "Array initialization error" << endl;
        return nullptr;
    }
    return arr;
}
```
- **Host Memory Allocation**: Uses `new(nothrow)` to allocate memory safely on the CPU without throwing exceptions.
- **Null Check**: Returns `nullptr` if allocation fails, ensuring safe handling.

#### 3. Error Handling - `checkCudaError`
```cpp
void checkCudaError(cudaError_t err, const char *operation) {
    if (err != cudaSuccess) {
        cout << "CUDA Error in operation: " << operation << " --- Error: "
             << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}
```
- **Purpose**: Checks the return value of CUDA API calls for errors.
- **Output**: Prints a descriptive error message (e.g., "cudaMalloc d_a failed") and exits if an error occurs.
- **Usage**: Applied after every CUDA call to ensure reliability.

#### 4. Synchronization & Error Checking
```cpp
checkCudaError(cudaGetLastError(), "kernel launch");
checkCudaError(cudaDeviceSynchronize(), "kernel execution");
```
- `cudaGetLastError()`: Checks for errors immediately after kernel launch (e.g., invalid grid/block sizes).
- `cudaDeviceSynchronize()`: Waits for all kernel threads to complete, ensuring no race conditions or incomplete execution before proceeding.

#### 5. Grid & Block Size Calculation
```cpp
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;
```
- **Block Size**: Set to `256` threads per block (a common choice for good performance).
- **Grid Size**: Calculated as `(N + block_size - 1) / block_size` to ensure enough blocks to cover all `N` elements, even if `N` isn’t perfectly divisible by `block_size`.

---

## Safe Practices Used

### Null Pointer Check Before Accessing Memory
```cpp
if (arr == nullptr) {
    cout << "Array initialization error" << endl;
    return nullptr;
}
```
- Prevents dereferencing null pointers by checking allocation success.

### Safe Memory Deallocation
```cpp
delete[] A;
delete[] B;
delete[] C;
A = nullptr;
B = nullptr;
C = nullptr;
```
- Frees host memory to prevent memory leaks.
- Sets pointers to `nullptr` to avoid dangling pointer issues.

### Error Handling in Memory Allocation
```cpp
if (C == nullptr) {
    delete[] A;
    delete[] B;
    cout << "Failed to initialize array C" << endl;
    return 1;
}
```
- Gracefully handles allocation failures by cleaning up and exiting.

---

## Output Example
```
0 2 4 6 8 10 12 14 16 18 ...
```
- Each element in `C[i] = A[i] + B[i]` is computed correctly and printed.

---

## Key Takeaways
- **CUDA Error Checking**: Ensured robustness by checking every CUDA call.
- **Memory Management**: Properly allocated and freed memory on both CPU and GPU.
- **Error Handling**: Added safeguards for null pointers and allocation failures.
- **Synchronization**: Used `cudaDeviceSynchronize()` to ensure correct execution order.

