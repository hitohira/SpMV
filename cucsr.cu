#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "matrix.h"

const int warpSize = 32;
const int NUM_VECTORS_PER_BLOCK = 1024;


__device__ int vecGetRowIndex(const int V,int* row_counter){
	int vecLaneId = threadIdx.x % V;
	int row = 0;
	if(vecLaneId == 0){
		row = atomicAdd(row_counter,1);
	}
	return __shfl(row,0,V);
}
__device__ int warpGetRowIndex(const int V,int* row_counter){
	int laneIdInWarp = threadIdx.x & (warpSize-1);
	int vecIdInWarp = laneIdInWarp / V;
	int row = 0;
	if(laneIdInWarp == 0){
		row = atomicAdd(row_counter,warpSize / V);
	}
	return __shfl(row,0,warpSize) + vecIdInWarp;
}
__global__ void spmvKernelS(const int V,int* row_counter,const int R,const int* row_offset,
                            const int* col,const float* val, const float* b,float* c){
	int vecLaneId = threadIdx.x % V;
	int vectorId = threadIdx.x / V;
	__shared__ volatile int shrdMem[NUM_VECTORS_PER_BLOCK][2];
	int row = warpGetRowIndex(V,row_counter);

	while(row < R){
		if(vecLaneId < 2){
			shrdMem[vectorId][vecLaneId] = row_offset[row+vecLaneId];
		}
		int row_s = shrdMem[vectorId][0];
		int row_e = shrdMem[vectorId][1];
		float dot_prod = 0.0;
		if(V == warpSize){
			int i = row_s - (row_s & (V-1)) + vecLaneId;
			if(i >= row_s && i < row_e){
				dot_prod += val[i] * b[col[i]];
			}
			for(i+=V; i < row_e; i+=V){
				dot_prod += val[i] * b[col[i]];
			}
		}
		else{
			for(int i = row_s + vecLaneId; i < row_e; i+=V){
				dot_prod += val[i] * b[col[i]];
			}
		}
		for(int i = V>>1; i > 0; i>>=1){
			dot_prod += __shfl_down(dot_prod,i,V);
		}
		if(vecLaneId == 0){
			c[row] = dot_prod;
		}
		row = warpGetRowIndex(V,row_counter);
	}
}
__global__ void spmvKernelD(const int V,int* row_counter,const int R,const int* row_offset,
                            const int* col,const double* val, const double* b,double* c){
}

template<typename X>
void wrapKernel(const int B,const int T,const int V,int* row_counter,const int R,const int* row_offset,
                const int* col,const X* val, const X* b,X* c){
	return;
}

template<> void wrapKernel(const int B, const int T,const int V,int* row_counter,const int R,const int* row_offset,
                const int* col,const float* val, const float* b,float* c){
	spmvKernelS<<<B,T>>>(V,row_counter,R,row_offset,col,val,b,c);
}
template<> void wrapKernel(const int B,const int T,const int V,int* row_counter,const int R,const int* row_offset,
                const int* col,const double* val, const double* b,double* c){
	spmvKernelD<<<B,T>>>(V,row_counter,R,row_offset,col,val,b,c);
}


template<typename T>
void CSR<T>::copyMatToDevice(T** d_val,int** d_rowptr,int** d_colind){
	int nnz = rowptr[m];
	
	if(*d_val) cudaFree(*d_val);
	if(*d_rowptr) cudaFree(*d_rowptr);
	if(*d_colind) cudaFree(*d_colind);
	cudaMalloc((void**)d_val,nnz*sizeof(T));
	cudaMalloc((void**)d_colind,nnz*sizeof(int));
	cudaMalloc((void**)d_rowptr,(m+1)*sizeof(int));

	cudaMemcpy(*d_val,val,nnz*sizeof(T),cudaMemcpyHostToDevice);
	cudaMemcpy(*d_colind,colind,nnz*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(*d_rowptr,rowptr,(m+1)*sizeof(int),cudaMemcpyHostToDevice);
}
template void CSR<float>::copyMatToDevice(float** d_val,int** d_rowptr,int** d_colind);
template void CSR<double>::copyMatToDevice(double** d_val,int** d_rowptr,int** d_colind);

template<typename T>
void Vec<T>::allocVectorToDevice(T** d_v){
	if(*d_v) cudaFree(*d_v);
	cudaMalloc((void**)d_v,m*sizeof(T));
}
template void Vec<float>::allocVectorToDevice(float** d_v);
template void Vec<double>::allocVectorToDevice(double** d_v);

template<typename T>
void Vec<T>::setVectorValueToDevice(T* d_v){
	cudaMemcpy(d_v,val,m*sizeof(T),cudaMemcpyHostToDevice);
}
template void Vec<float>::setVectorValueToDevice(float* d_v);
template void Vec<double>::setVectorValueToDevice(double* d_v);

template<typename X>
int gpuLightSpMV(CSR<X>& csr,Vec<X>& x,Vec<X>& y){
	if(csr.n != x.m || csr.m != y.m){
		fprintf(stderr,"wrong size of data\n");
		return -1;
	}
	int m = csr.m;
	int n = csr.n;
	int nnz = csr.rowptr[m];

	X* csr_val = NULL;
	int* csr_row = NULL;
	int* csr_col = NULL;
	X* b_val = NULL;
	X* c_val = NULL;
	csr.copyMatToDevice(&csr_val,&csr_row,&csr_col);
	x.allocVectorToDevice(&b_val);
	y.allocVectorToDevice(&c_val);
	x.setVectorValueToDevice(b_val);

	int* row_counter;
	cudaMalloc((void**)&row_counter,sizeof(int));
	cudaMemset(row_counter,0,sizeof(int));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	int T = prop.maxThreadsPerBlock;
	int B = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / T;
	int avgRowLength = nnz / m;
	
	printf("B=%d T=%d ave=%d\n",B,T,avgRowLength);

	// Kernel
/*
	cudaEvent_t e_s,e_e;
	cudaEventCreate(&e_s);
	cudaEventCreate(&e_e);
	cudaEventRecord(e_s);
*/
	if(avgRowLength <= 2){
		wrapKernel(B,T,2,row_counter,m,csr_row,csr_col,csr_val,b_val,c_val);
	}
	else if(avgRowLength <= 4){
		wrapKernel(B,T,4,row_counter,m,csr_row,csr_col,csr_val,b_val,c_val);
	}
	else if(avgRowLength <= 64){
		wrapKernel(B,T,8,row_counter,m,csr_row,csr_col,csr_val,b_val,c_val);
	}
	else{
		wrapKernel(B,T,32,row_counter,m,csr_row,csr_col,csr_val,b_val,c_val);
	}
/*
	cudaEventRecord(e_e);
	cudaEventSynchronize(e_e);
	float elasped = 0;
	cudaEventElaspedTime(&elasped,e_s,e_e);
	elasped /= 1000.0f;
	fprintf(stderr,"LightSpMV Kernel Exec Time : %f sec\n",elasped);
	fprintf(stderr,"LightSpMV Kernel FLOPS : %f MFLOPS\n",2*nnz/elasped*1e-6);
*/
	// end Kernel

	cudaError_t err = cudaMemcpy(y.val,c_val,n*sizeof(X),cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		fprintf(stderr,"spmv copy y: error code %d\n",err);
	}
	
	cudaFree(row_counter);
	cudaFree(c_val);
	cudaFree(b_val);
	cudaFree(csr_row);
	cudaFree(csr_col);
	cudaFree(csr_val);
	return 0;
}
template int gpuLightSpMV(CSR<float>& csr,Vec<float>& x,Vec<float>& y);
template int gpuLightSpMV(CSR<double>& csr,Vec<double>& x,Vec<double>& y);
