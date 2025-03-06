
---

# **CUDA 100-Day Challenge - Day 1**

### **Today's Task**  
Implemented a **basic vector addition** using CUDA, ensuring proper memory allocation, error handling, and synchronization.

---

## **Code Explanation**
### **1. Kernel Function - `vectorAddition`**
```cpp
__global__ void vectorAddition(float *A, float *B, float *C, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        *(C + idx) = *(B + idx) + *(A + idx);
    }
}
```
- Each thread computes `C[idx] = A[idx] + B[idx]`.
- **Index Calculation**:  
  - `blockDim.x` → Number of threads per block  
  - `blockIdx.x` → Current block index  
  - `threadIdx.x` → Current thread index within the block  
  - The final index is `idx = blockDim.x * blockIdx.x + threadIdx.x`.  

### **2. Memory Allocation & Initialization**
```cpp
checkCudaError(cudaMalloc(&d_a, N * sizeof(float)), "cudaMalloc d_a");
```
- Allocates memory for **device (GPU) arrays**.
- Uses `sizeof(float)` to ensure the correct byte size.

```cpp
float *Initialize(int N) {
    float *arr = new(nothrow) float[N];  // Safe memory allocation
}
```
- Uses `new(nothrow)` to avoid exceptions on allocation failure.

### **3. Error Handling (`checkCudaError`)**
```cpp
void checkCudaError(cudaError_t err, const char *operation) {
    if(err != cudaSuccess) {
        cout << "CUDA Error in operation: " << operation << " --- Error: " 
             << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}
```
- **Checks for CUDA errors** after every CUDA function call.
- Prints an error message and exits if a CUDA function fails.

### **4. Synchronization & Error Checking**
```cpp
checkCudaError(cudaGetLastError(), "kernel launch");
checkCudaError(cudaDeviceSynchronize(), "kernel execution");
```
- **`cudaGetLastError()`** detects errors from **kernel launch**.
- **`cudaDeviceSynchronize()`** ensures all **kernel executions are completed** before moving forward.

### **5. Grid & Block Size Calculation**
```cpp
int block_size = 256;
int grid_size = (N + block_size - 1) / block_size;
```
- Ensures enough **blocks** are created to process all `N` elements.

---

## **Safe Practices Used**
1. **Null Pointer Check Before Accessing Memory**
   ```cpp
   if (arr == nullptr) {
       cout << "Array initialization error" << endl;
       return nullptr;
   }
   ```
   - Prevents **dereferencing null pointers**.
   - Ensures safe memory allocation.

2. **Safe Memory Deallocation**
   ```cpp
   delete [] A;
   delete [] B;
   delete [] C;
   A = nullptr;
   B = nullptr;
   C = nullptr;
   ```
   - Prevents **memory leaks** by freeing host memory.
   - **Sets pointers to `nullptr`** to avoid dangling pointers.

3. **Error Handling in Memory Allocation**
   ```cpp
   if (C == nullptr) {
       delete [] A;
       delete [] B;
       cout << "Failed to initialize array C" << endl;
       return 1;
   }
   ```
   - Ensures **graceful exit** if memory allocation fails.

---

## **Output Example**
```
0 2 4 6 8 10 12 14 16 18 ...
```
(Each element in `C[i] = A[i] + B[i]`)

---

## **Key Takeaways**
**Used CUDA Error Checking** after each CUDA call.  
**Ensured Proper Memory Management** (both CPU and GPU).  
**Handled Potential Errors** (null pointer checks, memory allocation failures).  
**Implemented Safe Practices** (synchronization, freeing memory properly).  

--- 
