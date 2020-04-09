#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>

template<typename T>
class Vec{
public:
	int m = 0;
	T* val = NULL;
	~Vec(){
		if(val) free(val);
	}
	void Dump(){
		printf("Dump Vec\nN = %d\n",m);
			for(int i = 0; i < m; i++){
				printf("%f ",val[i]);
			}
			puts("END Dump");
		};
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
	void MulOnCPU(Vec<T> x, Vec<T> y);
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
	void MulOnCPU(Vec<T> x, Vec<T> y);
	void MulOnGPU(Vec<T> x, Vec<T> y);
};



#endif
