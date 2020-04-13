#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

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
__global__ void spmv_kernel_d(const int V,int* row_counter,const int R,const int* row_offset,
                            const int* col,const double* val, const double* b,double* c){
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
		double dot_prod = 0.0;
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

template<typename T>
int gpuLightSpMV(CSR<T>& csr,Vec<T>& x,Vec<T>& y){
	if(csr.s != x.m || csr.n != y.m){
		fprintf(stderr,"wrong size of data\n");
		return -1;
	}
	int m = csr.m;
	int n = csr.n;
	int nnz = csr.rowptr[m];

	T* csr_val;
	int* csr_row;
	int* csr_col;
	T* b_val;
	T* c_val;
	cudaMalloc((void**)&csr_val,nnz*sizeof(T));
	cudaMalloc((void**)&csr_col,nnz*sizeof(int));
	cudaMalloc((void**)&csr_row,(m+1)*sizeof(int));
	cudaMalloc((void**)&b_val,m*sizeof(T));
	cudaMalloc((void**)&c_val,n*sizeof(T));

	int* row_counter;
	cudaMalloc((void**)&row_counter,sizeof(int));
	cudaMemset(row_counter,0,sizeof(int));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	int T = prop.maxThreadsPerBlock;
	int B = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor / T;
	int avgRowLength = nnz / m;
	
	printf("B=%d T=%d ave=%d\n",B,T,avgRowLength);

	cudaMemcpy(b_val,x.val,x.m*sizeof(T),cudaMemcpyHostToDevice);

	cudaMemcpy(csr_val,csr.val,nnz*sizeof(T),cudaMemcpyHostToDevice);
	cudaMemcpy(csr_col,csr.colind,nnz*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(csr_row,csr.rowptr,(m+1)*sizeof(int),cudaMemcpyHostToDevice)


	// Kernel
/*
	cudaEvent_t e_s,e_e;
	cudaEventCreate(&e_s);
	cudaEventCreate(&e_e);
	cudaEventRecord(e_s);
*/
	if(avgRowLength <= 2){
		spmv_kernel<<<B,T>>>(2,row_counter,m,csr_row,csr_col,csr_val,b_val,c_val);
	}
	else if(avgRowLength <= 4){
		spmv_kernel<<<B,T>>>(4,row_counter,m,csr_row,csr_col,csr_val,b_val,c_val);
	}
	else if(avgRowLength <= 64){
		spmv_kernel<<<B,T>>>(8,row_counter,m,csr_row,csr_col,csr_val,b_val,c_val);
	}
	else{
		spmv_kernel<<<B,T>>>(32,row_counter,m,csr_row,csr_col,csr_val,b_val,c_val);
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

	cudaError_t err = cudaMemcpy(y.val,c_val,n*sizeof(T),cudaMemcpyDeviceToHost);
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
