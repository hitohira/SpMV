#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix.h"

template<typename T>
void FreeDeviceMemory(T* d_ptr){
	cudaFree(d_ptr);
}
template void FreeDeviceMemory(float* d_ptr);
template void FreeDeviceMemory(double* d_ptr);
template void FreeDeviceMemory(int* d_ptr);


template<typename T>
int Vec<T>::AllocVectorToDevice(){
	if(d_val) cudaFree(d_val);
	if(cudaMalloc((void**)&d_val,m*sizeof(T)) == cudaSuccess){
		return 0;
	}
	else{
		return -1;
	}
}
template int Vec<float>::AllocVectorToDevice();
template int Vec<double>::AllocVectorToDevice();

template<typename T>
int Vec<T>::SetVectorValueToDevice(){
	if(cudaMemcpy(d_val,val,m*sizeof(T),cudaMemcpyHostToDevice) == cudaSuccess){
		return 0;
	}
	return -1;
}
template int Vec<float>::SetVectorValueToDevice();
template int Vec<double>::SetVectorValueToDevice();

template<typename T>
int Vec<T>::GetVectorValueFromDevice(){
	if(cudaMemcpy(val,d_val,m*sizeof(T),cudaMemcpyDeviceToHost) == cudaSuccess){
		return 0;
	}
	return -1;
}
template int Vec<float>::GetVectorValueFromDevice();
template int Vec<double>::GetVectorValueFromDevice();



template<typename T>
int CSR<T>::CopyMatToDevice(){
		cudaError_t err;
	  int nnz = rowptr[m];
		if(d_val) cudaFree(d_val);
		if(d_rowptr) cudaFree(d_rowptr);
		if(d_colind) cudaFree(d_colind);
		 err = cudaMalloc((void**)&d_val,nnz*sizeof(T));
		 if(err != cudaSuccess){ fprintf(stderr,"fail to malloc on GPU\n");return -1;}
		 err = cudaMalloc((void**)&d_colind,nnz*sizeof(int));
		 if(err != cudaSuccess){ fprintf(stderr,"fail to malloc on GPU\n");return -1;}
		 err = cudaMalloc((void**)&d_rowptr,(m+1)*sizeof(int));
		 if(err != cudaSuccess){ fprintf(stderr,"fail to malloc on GPU\n");return -1;}
		 err = cudaMemcpy(d_val,val,nnz*sizeof(T),cudaMemcpyHostToDevice);
		 if(err != cudaSuccess){ fprintf(stderr,"fail to memcpy to GPU\n");return -1;}
		 err = cudaMemcpy(d_colind,colind,nnz*sizeof(int),cudaMemcpyHostToDevice);
		 if(err != cudaSuccess){ fprintf(stderr,"fail to memcpy to GPU\n");return -1;}
		 err = cudaMemcpy(d_rowptr,rowptr,(m+1)*sizeof(int),cudaMemcpyHostToDevice);
		 if(err != cudaSuccess){ fprintf(stderr,"fail to memcpy to GPU\n");return -1;}
		 return 0;
}
template int CSR<float>::CopyMatToDevice();
template int CSR<double>::CopyMatToDevice();


template<typename T>
int ELL<T>::CopyMatToDevice(){
	cudaError_t err;
	if(d_val) cudaFree(d_val);
	if(d_colind) cudaFree(d_colind);
	err = cudaMalloc((void**)&d_val,m*k*sizeof(T));
	if(err != cudaSuccess){ fprintf(stderr,"fail to malloc on GPU\n");return -1;}
	err = cudaMalloc((void**)&d_colind,m*k*sizeof(int));
	if(err != cudaSuccess){ fprintf(stderr,"fail to malloc on GPU\n");return -1;}

	err = cudaMemcpy(d_val,val,m*k*sizeof(T),cudaMemcpyHostToDevice);
	if(err != cudaSuccess){ fprintf(stderr,"fail to memcpy to GPU\n");return -1;}
	err = cudaMemcpy(d_colind,colind,m*k*sizeof(T),cudaMemcpyHostToDevice);
	if(err != cudaSuccess){ fprintf(stderr,"fail to memcpy to GPU\n");return -1;}
	return 0;
}
template int ELL<float>::CopyMatToDevice();
template int ELL<double>::CopyMatToDevice();

template<typename T>
int COO<T>::CopyMatToDevice(){
	cudaError_t err;
	if(d_val) cudaFree(d_val);
	if(d_colind) cudaFree(d_colind);
	if(d_rowind) cudaFree(d_rowind);
	err = cudaMalloc((void**)&d_val,nnz*sizeof(T));
	if(err != cudaSuccess) {fprintf(stderr,"fail at malloc on GPU\n");return -1;}
	err = cudaMalloc((void**)&d_colind,nnz*sizeof(T));
	if(err != cudaSuccess) {fprintf(stderr,"fail at malloc on GPU\n");return -1;}
	err = cudaMalloc((void**)&d_rowind,nnz*sizeof(T));
	if(err != cudaSuccess) {fprintf(stderr,"fail at malloc on GPU\n");return -1;}

	err = cudaMemcpy(d_val,val,nnz*sizeof(T),cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { fprintf(stderr,"fail at memcpy to GPU\n");return -1;}
	err = cudaMemcpy(d_colind,colind,nnz*sizeof(T),cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { fprintf(stderr,"fail at memcpy to GPU\n");return -1;}
	err = cudaMemcpy(d_rowind,rowind,nnz*sizeof(T),cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { fprintf(stderr,"fail at memcpy to GPU\n");return -1;}
	return 0;
}
template int COO<float>::CopyMatToDevice();
template int COO<double>::CopyMatToDevice();
