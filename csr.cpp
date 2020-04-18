#include "matrix.h"
#include <mkl.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>

template<typename T>
int fscanWrap(FILE* fp,int* d1,int* d2,T* f1){
	return -1;
}
template<>
int fscanWrap(FILE* fp,int* d1,int* d2,float* f1){
	return fscanf(fp,"%d %d %f",d1,d2,f1);
}
template<>
int fscanWrap(FILE* fp,int* d1,int* d2,double* f1){
	return fscanf(fp,"%d %d %lf",d1,d2,f1);
}



template<typename T>
void tCSRLoadFromMM(const char* fileName,CSR<T>* tcsr){

	static char buf[4096];
	FILE* fp;
	fp = fopen(fileName,"r");
	if(fp == NULL) return;
	// skip comment
	while(1){
		if(fgets(buf,4096,fp) == NULL){
			fclose(fp);
			return;	
		}
		if(buf[0] != '%'){
			break;
		}
	}
	int nz;
	sscanf(buf,"%d %d %d",&(tcsr->n),&(tcsr->m),&nz); // because transposed m<->n
	

	tcsr->val = (T*)malloc(nz*sizeof(T));
	tcsr->colind = (int*)malloc(nz*sizeof(int));
	tcsr->rowptr = (int*)malloc(((tcsr->m)+1)*sizeof(int));
	tcsr->rowptr[tcsr->m] = nz;

	int tmpR = -1;
	for(int i = 0; i < nz; i++){
		if(i % 1000000 == 0) fprintf(stderr,"%d / %d\n",i,nz);
		int x,y;
		T f;
		if(fscanWrap(fp,&y,&x,&f) != 3){
			fclose(fp);
			tcsr->m = tcsr->n = 0;
			return;
		}
		tcsr->val[i] = f;
		tcsr->colind[i] = y-1;
		if(x-1 != tmpR){
			while(tmpR != x-1){
				tmpR++;
				tcsr->rowptr[tmpR] = 0;
			}
			tcsr->rowptr[x-1] = i;
			tmpR = x-1;
		}
	}
	fclose(fp);
}
template void tCSRLoadFromMM(const char* fileName,CSR<float>* tcsr);
template void tCSRLoadFromMM(const char* fileName,CSR<double>* tcsr);

template<typename T>
void CSR<T>::LoadFromMM(const char* fileName){
	tCSRLoadFromMM<T>(fileName,this);
	Transpose();
}
template void CSR<float>::LoadFromMM(const char* fileName);
template void CSR<double>::LoadFromMM(const char* fileName);

template<typename T>
void CSR<T>::Transpose(){
	std::vector<std::vector<std::pair<int,T> > > vlist(n);
	for(int i = 0; i < m; i++){
		for(int j = rowptr[i]; j < rowptr[i+1]; j++){
			vlist[colind[j]].push_back(std::make_pair(i,val[j]));
		}
	}
	int newN = m;
	int newM = n;
	int nnz = rowptr[m];
	free(rowptr);
	rowptr = (int*)malloc((newM+1)*sizeof(int));
	if(rowptr == NULL) {
		fprintf(stderr,"malloc error\n");
		return;
	}
	int cntr = 0;
	rowptr[newM] = nnz;
	for(int i = 0; i < newM; i++){
		rowptr[i] = cntr;
		for(int j = 0; j < vlist[i].size(); j++){
			colind[cntr] = vlist[i][j].first;
			val[cntr] = vlist[i][j].second;
			cntr++;
		}
	}
	if(cntr != nnz) fprintf(stderr,"not match nnz\n");
}
template void CSR<float>::Transpose();
template void CSR<double>::Transpose();

template<typename T>
void CSR<T>::Dump(){
	printf("dump CSR\n M * N = %d * %d\n",m,n);
	puts("value");
	for(int i = 0; i < m; i++){
		for(int j = rowptr[i]; j < rowptr[i+1]; j++){
			printf("%f ",val[j]);
		}
		puts("");
	}
	puts("colind");
	for(int i = 0; i < m; i++){
		for(int j = rowptr[i]; j < rowptr[i+1]; j++){
			printf("%d ",colind[j]);
		}
		puts("");
	}
	printf("END Dump");
}
template void CSR<float>::Dump();
template void CSR<double>::Dump();

template<typename T>
void CSR<T>::MulOnCPU(Vec<T>& x, Vec<T>& y){
	int i,j;
	#pragma omp parallel for private(j) 
	for(i = 0; i < m; i++){
		y.val[i] = 0;
		for(j = rowptr[i]; j < rowptr[i+1]; j++){
			y.val[i] += val[j] * x.val[colind[j]];
		}
	}	
}
template void CSR<float>::MulOnCPU(Vec<float>& x, Vec<float>& y);
template void CSR<double>::MulOnCPU(Vec<double>& x, Vec<double>& y);


template<typename T>
void CSR<T>::MklMul(Vec<T>& x, Vec<T>& y){
	fprintf(stderr,"MklMul : Wrong Type of CSR\n");
	return;
}
template<> void CSR<float>::MklMul(Vec<float>& x, Vec<float>& y){
	char ts = 'N';
	mkl_cspblas_scsrgemv (
		&ts , 
		&m , 
		val , 
		rowptr , 
		colind , 
		x.val , 
		y.val );	
}
template<> void CSR<double>::MklMul(Vec<double>& x, Vec<double>& y){
	char ts = 'N';
	mkl_cspblas_dcsrgemv (
		&ts , 
		&m , 
		val , 
		rowptr , 
		colind , 
		x.val , 
		y.val );	
}

