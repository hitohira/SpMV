#include <stdio.h>
#include <mkl.h>
#include <omp.h>
#include <stdlib.h>
#include "matrix.h"

template<typename T>
void COO<T>::TransformFromCSR(const CSR<T>& csr){
	m = csr.m;
	n = csr.n;
	nnz = csr.rowptr[m];
	if(val) free(val);
	if(colind) free(colind);
	if(rowind) free(rowind);
	val = (T*)malloc(nnz*sizeof(T));
	colind = (int*)malloc(nnz*sizeof(T));
	rowind = (int*)malloc(nnz*sizeof(T));
	if(val == NULL || colind == NULL || rowind == NULL) return;
	#pragma omp parallel for
	for(int i = 0; i < nnz; i++){
		val[i] = csr.val[i];
	}
	#pragma omp parallel for
	for(int i = 0; i < nnz; i++){
		colind[i] = csr.colind[i];
	}
	int idx = 0;
	for(int i = 0; i < m; i++){
		for(int j = csr.rowptr[i]; j < csr.rowptr[i+1]; j++){
			rowind[idx++] = i;
		}
	}
}
template void COO<float>::TransformFromCSR(const CSR<float>& csr);
template void COO<double>::TransformFromCSR(const CSR<double>& csr);

template <typename T>
void COO<T>::Dump(){
	printf("Dump COO\nM * N = %d * %d, nnz = %d\n",m,n,nnz);
	puts("value");
	for(int i = 0; i < nnz; i++){
		printf("%f ",val[i]);
	}
	puts("");
	puts("colind");
	for(int i = 0; i < nnz; i++){
		printf("%f ",colind[i]);
	}
	puts("");
	puts("rowind");
	for(int i = 0; i < nnz; i++){
		printf("%f ",rowind[i]);
	}
	puts("");
	puts("END Dump");
}
template void COO<float>::Dump();
template void COO<double>::Dump();

template<typename T>
void COO<T>::MulOnCPU(Vec<T>& x,Vec<T>& y){
	fprintf(stderr,"wrong Type\n");
}
template<> void COO<float>::MulOnCPU(Vec<float>& x,Vec<float>& y){
	char ts = 'N';
	mkl_cspblas_scoogemv (&ts,&m,val,rowind,colind,&nnz,x.val,y.val);
}
template<> void COO<double>::MulOnCPU(Vec<double>& x,Vec<double>& y){
	char ts = 'N';
	mkl_cspblas_dcoogemv (&ts,&m,val,rowind,colind,&nnz,x.val,y.val);
}
