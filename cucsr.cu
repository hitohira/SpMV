#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
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
	return __shfl_sync(0xffffffff,row,0,V);
//	return __shfl(row,0,V);
}
__device__ int warpGetRowIndex(const int V,int* row_counter){
	int laneIdInWarp = threadIdx.x & (warpSize-1);
	int vecIdInWarp = laneIdInWarp / V;
	int row = 0;
	if(laneIdInWarp == 0){
		row = atomicAdd(row_counter,warpSize / V);
	}
	return __shfl_sync(0xffffffff,row,0,warpSize) + vecIdInWarp;
//	return __shfl(row,0,warpSize) + vecIdInWarp;
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
			dot_prod += __shfl_down_sync(0xffffffff,dot_prod,i,V);
//			dot_prod += __shfl_down(dot_prod,i,V);
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


template<typename X>
void CSR<X>::MulLightSpMVOnGPU(Vec<X>& x,Vec<X>& y){
	if(n != x.m || m != y.m){
		fprintf(stderr,"wrong size of data\n");
		return;
	}
	int nnz = rowptr[m];

/*
	CopyMatToDevice();
	x.AllocVectorToDevice();
	y.AllocVectorToDevice();
	x.SetVectorValueToDevice();
	y.Fill(0);
	y.SetVectorValueToDevice();
*/
	cudaDeviceProp prop;
	if(cudaGetDeviceProperties(&prop,0) != cudaSuccess){
		fprintf(stderr,"fail at accessing GPU device\n");
		return;
	}

	int T = prop.maxThreadsPerBlock;
	int B = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / T;
	int avgRowLength = nnz / m;
	
//	fprintf(stderr,"B=%d T=%d ave=%d\n",B,T,avgRowLength);

	int* row_counter;
	cudaMalloc((void**)&row_counter,sizeof(int));
	cudaMemset(row_counter,0,sizeof(int));

	// Kernel
/*
	cudaEvent_t e_s,e_e;
	cudaEventCreate(&e_s);
	cudaEventCreate(&e_e);
	cudaEventRecord(e_s);
*/
	if(avgRowLength <= 2){
		wrapKernel(B,T,2,row_counter,m,d_rowptr,d_colind,d_val,x.d_val,y.d_val);
	}
	else if(avgRowLength <= 4){
		wrapKernel(B,T,4,row_counter,m,d_rowptr,d_colind,d_val,x.d_val,y.d_val);
	}
	else if(avgRowLength <= 64){
		wrapKernel(B,T,8,row_counter,m,d_rowptr,d_colind,d_val,x.d_val,y.d_val);
	}
	else{
		wrapKernel(B,T,32,row_counter,m,d_rowptr,d_colind,d_val,x.d_val,y.d_val);
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
	
	y.GetVectorValueFromDevice();
	
	cudaFree(row_counter);
	return;
}
template void CSR<float>::MulLightSpMVOnGPU(Vec<float>& x,Vec<float>& y);
template void CSR<double>::MulLightSpMVOnGPU(Vec<double>& x,Vec<double>& y);



// see toolkit document v10.0
template<typename T>
void CSR<T>::CuSparseMul(Vec<T>& x, Vec<T>& y){
}
template<> void CSR<float>::CuSparseMul(Vec<float>& x,Vec<float>& y){
	if(d_val == NULL) return;
	cusparseHandle_t handle;
	cusparseMatDescr_t descr;

	float alpha = 1;
	float beta = 0;

	cusparseCreate(&handle);
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m,n,rowptr[m],
	                &alpha,descr,d_val,d_rowptr,d_colind,x.d_val,&beta,y.d_val);

	cusparseDestroyMatDescr(descr);
	cusparseDestroy(handle);
}
template<> void CSR<double>::CuSparseMul(Vec<double>& x,Vec<double>& y){
	if(d_val == NULL) return;
	cusparseHandle_t handle;
	cusparseMatDescr_t descr;

	double alpha = 1;
	double beta = 0;

	cusparseCreate(&handle);
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m,n,rowptr[m],
	                &alpha,descr,d_val,d_rowptr,d_colind,x.d_val,&beta,y.d_val);

	cusparseDestroyMatDescr(descr);
	cusparseDestroy(handle);
}
