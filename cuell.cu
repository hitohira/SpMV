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
			dot += v * b[idx];
		}
		c[r] = dot;
	}
}


template<typename X>
void ELL<X>::MulOnGPU(Vec<X>& x, Vec<X>& y){
}
template<> void ELL<float>::MulOnGPU(Vec<float>& x, Vec<float>& y){
	int T = 1024;
	int B = m / T + 1;
	EllKernelS<<<B,T>>>(m,k,d_colind,d_val,x.d_val,y.d_val);
}
template<> void ELL<double>::MulOnGPU(Vec<double>& x, Vec<double>& y){

}


__global__ void EllKernelSWithTex(const int nrow,const int width,
                                const int* col,const float* val, cudaTextureObject_t b,float* c){
	int r = blockDim.x * blockIdx.x + threadIdx.x;
	if(r < nrow){
		float dot = 0;

		for(int i = 0; i < width; i++){
			int idx = col[nrow*i+r];
			float v = val[nrow*i+r];
			if(v != 0){
				float bval = tex1Dfetch<float>(b,idx);
				dot += v * bval;
			}
		}
		c[r] = dot;
	}
}


template<typename X>
void ELL<X>::MulOnGPUWithTex(Vec<X>& x, Vec<X>& y){
}
template<> void ELL<float>::MulOnGPUWithTex(Vec<float>& x, Vec<float>& y){
	int T = 1024;
	int B = m / T + 1;
	EllKernelSWithTex<<<B,T>>>(m,k,d_colind,d_val,*((cudaTextureObject_t*)(x.texVec.texObj)),y.d_val);
}
template<> void ELL<double>::MulOnGPUWithTex(Vec<double>& x, Vec<double>& y){

}

