#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

template<typename T>
class Vec{
public:
	int m = 0;
	T* val = NULL;
	~Vec(){
		if(val) free(val);
	}
	void Create(int m){
		this->m = m;
		val = (T*)malloc(m*sizeof(T));
	}
	void Fill(T x){
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
	bool Equal(Vec<T>& v){
		const T r = 1e-4;
		T nrm = Norm2();
		for(int i = 0; i < m; i++){
			if(fabs(val[i]-v.val[i]) > r * nrm){
				return false;
			}
		}
		return true;
	}
};

template<typename T>
class CSR{
public:
	int m = 0; // row
	int n = 0; // col
	T* val = NULL;
	int* colind = NULL;
	int* rowptr = NULL;
	~CSR(){
		printf("csr del\n");
		if(val) free(val);
		if(colind) free(colind);
		if(rowptr) free(rowptr);
	}
	void LoadFromMM(const char* filename);
	void Transpose();
	void Dump();
	void MulOnCPU(Vec<T>& x, Vec<T>& y);
};

template<typename T>
class ELL{
public:
	int m = 0; // row
	int n = 0; // col
	int k = 0;
	T* val = NULL;
	int* colind = NULL;
	~ELL(){
		printf("ell del\n");
		if(val) free(val);
		if(colind) free(colind);
	}
	void TransformFromCSR(const CSR<T>& csr);
	void Dump();
	void MulOnCPU(Vec<T>& x, Vec<T>& y);
	void MulOnGPU(Vec<T>& x, Vec<T>& y);
};



#endif
