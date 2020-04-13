#include "matrix.h"

int main(){
	CSR<float> csr;
	ELL<float> ell;
	Vec<float> x,x2;
	Vec<float> y,y2;
	printf("start\n");
//	csr.LoadFromMM("matrices/cage3/cage3.mtx");
	csr.LoadFromMM("matrices/ldoor/ldoor.mtx");
	ell.TransformFromCSR(csr);
	x.Create(csr.m);
	x.Fill(1.0f);
	x2.Copy(x);
	y.Create(csr.n);
	y.Fill(0);
	y2.Copy(y);
	
	csr.MulOnCPU(x,y);
	ell.MulOnCPU(x2,y2);
	if(y.Equal(y2)){
		printf("same result\n");
	}
	else{
		printf("result wrong\n");
	}
	printf("end\n");
	return 0;
}
