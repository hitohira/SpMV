#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

template<typename T>
void FreeDeviceMemory(T* d_ptr);

template<typename T>
class Vec{
public:
	int m;
	T* val;
	T* d_val;
	Vec(){
		m = 0;
		val = NULL;
		d_val = NULL;
	}
	~Vec(){
		if(val) free(val);
		if(d_val) FreeDeviceMemory(d_val);
	}
	void Create(int m){
		this->m = m;
		val = (T*)malloc(m*sizeof(T));
	}
	void Fill(T x){
		#pragma omp parallel for
		for(int i = 0; i < m; i++){
			val[i] = x;
		}
	}
	void Copy(Vec<T>& v){
		m = v.m;
		if(val != NULL) free(val);
		val = NULL;
		val = (T*)malloc(m*sizeof(T));
		if(val == NULL) return;
		#pragma omp parallel for
		for(int i = 0; i < m; i++){
			val[i] = v.val[i];
		}
	}
	void Dump(){
		printf("Dump Vec\nN = %d\n",m);
		for(int i = 0; i < m; i++){
			printf("%f ",val[i]);
		}
		puts("END Dump");
	};
	T Norm2(){
		T acc = 0;
		for(int i = 0; i < m; i++){
			acc += val[i] * val[i];
		}
		return (T)sqrt(acc);
	}
	bool Equal(Vec<T>& v,T eps){
		T nrm = Norm2();
		for(int i = 0; i < m; i++){
			if(fabs(val[i]-v.val[i]) > eps * nrm){
				fprintf(stderr,"Vec NormDiff[%d] = %f\n",i,(val[i] - v.val[i]) / nrm);
				return false;
			}
		}
		return true;
	}
	int AllocVectorToDevice();
	int SetVectorValueToDevice();
	int GetVectorValueFromDevice();
};

template<typename T>
class CSR{
public:
	int m; // row
	int n; // col
	T* val;
	int* colind;
	int* rowptr;
	T* d_val;
	int* d_colind;
	int* d_rowptr;
	CSR(){
		m = n = 0;
		val = NULL;
		colind = NULL;
		rowptr = NULL;
		d_val = NULL;
		d_colind = NULL;
		d_rowptr = NULL;
	}
	~CSR(){
		printf("csr del\n");
		if(val) free(val);
		if(colind) free(colind);
		if(rowptr) free(rowptr);
		if(d_val) FreeDeviceMemory(d_val);
		if(d_colind) FreeDeviceMemory(d_colind);
		if(d_rowptr) FreeDeviceMemory(d_rowptr);
	}
	void LoadFromMM(const char* filename);
	void Transpose();
	void Dump();
	void MulOnCPU(Vec<T>& x, Vec<T>& y);
	void MklMul(Vec<T>& x, Vec<T>& y);
	void MulOnGPU(Vec<T>& x, Vec<T>& y);
	void MulLightSpMVOnGPU(Vec<T>& x,Vec<T>& y);
	int CopyMatToDevice();
};

template<typename T>
class ELL{
public:
	int m; // row
	int n; // col
	int k;
	T* val;
	int* colind;
	T* d_val;
	int* d_colind;
	ELL(){
		m = n = k = 0;
		val = NULL;
		colind = NULL;
		d_val = NULL;
		d_colind = NULL;
	}
	~ELL(){
		printf("ell del\n");
		if(val) free(val);
		if(colind) free(colind);
		if(d_val) FreeDeviceMemory(d_val);
		if(d_colind) FreeDeviceMemory(d_colind);
	}
	void TransformFromCSR(const CSR<T>& csr);
	void Dump();
	void MulOnCPU(Vec<T>& x, Vec<T>& y);
	void MulOnGPU(Vec<T>& x, Vec<T>& y);
	int CopyMatToDevice();
};




#endif
