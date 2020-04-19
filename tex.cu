#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "tex.h"

template<typename T>
void TexVec<T>::Free(){
	if(cuArray){
		cudaFreeArray((cudaArray*)cuArray);
	}
	if(texObj){
		cudaDestroyTextureObject(*(cudaTextureObject_t*)texObj);
		free(texObj);
	}
}
template void TexVec<float>::Free();
template void TexVec<double>::Free();


template<typename T>
void TexVec<T>::SetTexVec(int m,T* data){
}
template<> void TexVec<float>::SetTexVec(int m,float* data){
	Free();
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
	int size = m * sizeof(float);
	int width = m;
	int height = 1;
	cudaMallocArray(((cudaArray**)(&cuArray)),&channelDesc,width,height);
	cudaMemcpyToArray((cudaArray*)cuArray,0,0,data,size,cudaMemcpyHostToDevice);

	struct cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = (cudaArray*)cuArray;

	struct cudaTextureDesc texDesc;
	memset(&texDesc,0,sizeof(texDesc));
	texDesc.addressMode[0]   = cudaAddressModeWrap;
	texDesc.addressMode[1]   = cudaAddressModeWrap;
	texDesc.filterMode       = cudaFilterModePoint;
	texDesc.readMode         = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	texObj = malloc(sizeof(cudaTextureObject_t));
	cudaCreateTextureObject((cudaTextureObject_t*)texObj,&resDesc,&texDesc,NULL);

}
template<> void TexVec<double>::SetTexVec(int m,double* data){
	Free();
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindSigned);

}
