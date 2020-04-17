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
void CSR<T>::CopyMatToDevice(){
	  int nnz = rowptr[m];
		if(d_val) cudaFree(d_val);
		if(d_rowptr) cudaFree(d_rowptr);
		if(d_colind) cudaFree(d_colind);
		 cudaMalloc((void**)&d_val,nnz*sizeof(T));
		 cudaMalloc((void**)&d_colind,nnz*sizeof(int));
		 cudaMalloc((void**)&d_rowptr,(m+1)*sizeof(int));
		 cudaMemcpy(d_val,val,nnz*sizeof(T),cudaMemcpyHostToDevice);
		 cudaMemcpy(d_colind,colind,nnz*sizeof(int),cudaMemcpyHostToDevice);
		 cudaMemcpy(d_rowptr,rowptr,(m+1)*sizeof(int),cudaMemcpyHostToDevice);
}
template void CSR<float>::CopyMatToDevice();
template void CSR<double>::CopyMatToDevice();

template<typename T>
void Vec<T>::AllocVectorToDevice(){
	if(d_val) cudaFree(d_val);
	cudaMalloc((void**)&d_val,m*sizeof(T));
}
template void Vec<float>::AllocVectorToDevice();
template void Vec<double>::AllocVectorToDevice();

template<typename T>
void Vec<T>::SetVectorValueToDevice(){
	cudaMemcpy(d_val,val,m*sizeof(T),cudaMemcpyHostToDevice);
}
template void Vec<float>::SetVectorValueToDevice();
template void Vec<double>::SetVectorValueToDevice();

template<typename T>
void Vec<T>::GetVectorValueFromDevice(){
	cudaMemcpy(val,d_val,m*sizeof(T),cudaMemcpyDeviceToHost);
}
template void Vec<float>::GetVectorValueFromDevice();
template void Vec<double>::GetVectorValueFromDevice();


template<typename T>
void ELL<T>::CopyMatToDevice(){
	if(d_val) cudaFree(d_val);
	if(d_colind) cudaFree(d_colind);
	cudaMalloc((void**)&d_val,m*k*sizeof(T));
	cudaMalloc((void**)&d_colind,m*k*sizeof(int));

	cudaMemcpy(d_val,val,m*k*sizeof(T),cudaMemcpyHostToDevice);
	cudaMemcpy(d_colind,colind,m*k*sizeof(T),cudaMemcpyHostToDevice);
}
template void ELL<float>::CopyMatToDevice();
template void ELL<double>::CopyMatToDevice();

