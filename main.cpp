#include "matrix.h"
#include "util.h"

const int TIMES = 10;

int main(){
	CSR<float> csr;
	ELL<float> ell;
	Vec<float> x,x2;
	Vec<float> y,y2;
	printf("start\n");
//	csr.LoadFromMM("matrices/cage3/cage3.mtx");
	csr.LoadFromMM("matrices/ldoor/ldoor.mtx");
//	csr.LoadFromMM("matrices/bbmat/bbmat.mtx");
	ell.TransformFromCSR(csr);
	x.Create(csr.n);
	x.Fill(1.0f);
	x2.Copy(x);
	y.Create(csr.m);
	y.Fill(0);
	y2.Copy(y);
	
	csr.MulOnCPU(x,y);
	csr.MklMul(x2,y2);
	double acc1 = 0;
	double acc2 = 0;
	for(int i = 0; i < TIMES; i++){
		double t1 = elasped();
		csr.MulOnCPU(x,y);
		double t2 = elasped();
		acc1 += t2-t1;
	}
	for(int i = 0; i < TIMES; i++){
		double t1 = elasped();
		csr.MklMul(x2,y2);
		double t2 = elasped();
		acc2 += t2-t1;
	}
//	ell.MulOnCPU(x2,y2);
	if(y.Equal(y2,1e-4)){
		printf("same result\n");
	}
	else{
		printf("result wrong\n");
	}
//	y.Dump();
//	y2.Dump();

	printf("elasped\nnaive : %f sec\nMKL : %f sec\n",acc1/10,acc2/10);
	printf("end\n");
	return 0;
}
