#include <mkl.h>
#include <stdio.h>

#include "matrix.h"


template<typename T>
void CSR<T>::MklMul(Vec<T>& x, Vec<T>& y){
	fprintf(stderr,"MklMul : Wrong Type of CSR\n");
	return;
}
template void CSR<float>::MklMul(Vec<float>& x, Vec<float>& y){
	mkl_cspblas_scsrgemv (
		'N' , 
		&m , 
		val , 
		rowptr , 
		colind , 
		x.val , 
		y.val );	
}
template void CSR<double>::MklMul(Vec<double>& x, Vec<double>& y){
	mkl_cspblas_dcsrgemv (
		'N' , 
		&m , 
		val , 
		rowptr , 
		colind , 
		x.val , 
		y.val );	
}

