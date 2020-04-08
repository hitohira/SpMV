#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

template<typename T>
void ELL<T>::TransformFromCSR(CSR<T> csr){
	m = csr.m;
	n = csr.n;
	int mx_col = 0;
	for(int i = 0; i < m; i++){
		int rnum = csr.rowptr[i+1] - csr.rowptr[i];
		mx_col = mx_col > rnum ? mx_col : rnum;
	}
	k = mx_col;
	if(val) free(val);
	if(colind) free(colind);
	val = (T*)malloc(m * k *sizeof(T));
	colind = (int*)malloc(m * k *sizeof(int));
	if(val == NULL || colind == NULL) return;
	for(int i = 0; i < m; i++){
		int tmp_col = 0;
		for(int j = csr.rowptr[i]; j < csr.rowptr[i+1]; j++){
			int idx = tmp_col * m + i;
			val[idx] = csr.val[j];
			colind[idx] = csr.colind[j];
			tmp_col++;
		}
		int rnum = csr.rowptr[i+1] - csr.rowptr[i];
		while(tmp_col < k){
			// fill 0
			val[tmp_col*m + i] = 0;
			tmp_col++;
		}
	}
}
template void ELL<float>::TransformFromCSR(CSR<float> csr);
template void ELL<double>::TransformFromCSR(CSR<double> csr);

template<typename T>
void ELL<T>::Dump(){
	printf("Dump ELL\nM * N = %d * %d, K = %d\n",m,n,k);
	puts("value");
	for(int i = 0; i < m; i++){
		for(int j = 0; j < k; j++){
			printf("%f ",val[i*m+j]);
		}
		puts("");
	}
	puts("colind");
	for(int i = 0; i < m; i++){
		for(int j = 0; j < k; j++){
			printf("%d ",colind[i*m+j]);
		}
		puts("");
	}
	puts("END Dump");
}
template void ELL<float>::Dump();
template void ELL<double>::Dump();

template<typename T>
void ELL<T>::MulOnCPU(Vec<T> x, Vec<T> y){
	int i,j;
	#pragma omp parallel for private(j) collapse(2)
	for(i = 0; i < m; i++){
		for(j = 0; j < k; j++){
			int idx = m*j + i;
			y.val[i] += val[idx] * x.val[colind[idx]];
		}
	}
}
template void ELL<float>::MulOnCPU(Vec<float> x, Vec<float> y);
template void ELL<double>::MulOnCPU(Vec<double> x, Vec<double> y);

template<typename T>
void ELL<T>::MulOnGPU(Vec<T> x, Vec<T> y){
	// TODO

}
template void ELL<float>::MulOnGPU(Vec<float> x, Vec<float> y);
template void ELL<double>::MulOnGPU(Vec<double> x, Vec<double> y);
