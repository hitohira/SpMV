#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include "matrix.h"


// see toolkit document v10.0
template<typename T>
void CSR<T>::CuSparseMul(Vec<T>& x, Vec<T>& y){
}
template<> void CSR<float>::CuSparseMul(Vec<float>& x,Vec<float>& y){
	cusparseHandle_t handle;
	cusparseMatDescr_t descr;

	float alpha = 1;
	float beta = 0;

	cusparseCreate(&handle);
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	cusparseScsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m,n,rowptr[m],
	                &alpha,descr,d_val,d_rowptr,d_colind,x.d_val,&beta,y.d_val);

	cusparseDestroyMatDescr(descr);
	cusparseDestroy(handle);
}
template<> void CSR<double>::CuSparseMul(Vec<double>& x,Vec<double>& y){
	cusparseHandle_t handle;
	cusparseMatDescr_t descr;

	double alpha = 1;
	double beta = 0;

	cusparseCreate(&handle);
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

	cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,m,n,rowptr[m],
	                &alpha,descr,d_val,d_rowptr,d_colind,x.d_val,&beta,y.d_val);

	cusparseDestroyMatDescr(descr);
	cusparseDestroy(handle);
}
