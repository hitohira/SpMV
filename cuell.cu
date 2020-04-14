#include <cuda.h>
#include <cuda_runtime.h>

#include "matrix.h"

__global__ void EllKernelS(const int nrow,const int width,
                                const int* col,const float* val, const float* b,float* c){
	int r = blockDim.x * blockIdx.x + threadIdx.x;
	if(r < nrow){
		float dot = 0;

		for(int i = 0; i < width; i++){
			int idx = col[nrow*i+r];
			float v = val[nrow*i+r];
			if(v != 0){
				dot += v * b[idx];
			}
		}
		c[r] = dot;
	}
}

template<typename T>
void ELL<T>::copyMatToDevice(T** d_val,int** d_colind){
	if(*d_val) cudaFree(*d_val);
	if(*d_colind) cudaFree(*d_colind);
	cudaMalloc((void**)d_val,m*k*sizeof(T));
	cudaMalloc((void**)d_colind,m*k*sizeof(int));

	cudaMemcpy(*d_val,val,m*k*sizeof(T),cudaMemcpyHostToDevice);
	cudaMemcpy(*d_colind,colind,m*k*sizeof(T),cudaMemcpyHostToDevice);
}

template<typename X>
void ELL<X>::MulOnGPU(Vec<X>& x, Vec<X>& y){
	int T = 1024;
	int B = m / T + 1;
	
}
template void ELL<float>::MulOnGPU(Vec<float>& x, Vec<float>& y);
template void ELL<double>::MulOnGPU(Vec<double>& x, Vec<double>& y);
