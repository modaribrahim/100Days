# GPU 100 Days Learning Journey

This document serves as a log of the progress and knowledge I gained while working on GPU programming and studying the **PMPP (Parallel Programming and Optimization)** book.

Idea by: https://github.com/hkproj/

---

## Day 1
### File: vectorAddition.cu
**Summary:**  
Implemented a basic vector addition using CUDA. Designed a kernel to perform parallel addition of two arrays, with each thread computing the sum of a pair of values. Added error handling, memory management, and synchronization for robustness.  

**Learned:**  
- Basics of writing and launching a CUDA kernel.  
- Grid and block size configuration for parallel execution.  
- Proper memory allocation and deallocation on both host (CPU) and device (GPU).  
- Importance of CUDA error checking and synchronization.  

### Reading:  
- Read **Chapter 1** of the PMPP book.  
  - Learned about the fundamentals of parallel programming, CUDA architecture, and the GPU execution model.

---