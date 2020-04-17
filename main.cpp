#include "matrix.h"
#include "util.h"

const int TIMES = 50;

int main(){
	CSR<float> csr;
	ELL<float> ell;
	Vec<float> x,x2;
	Vec<float> y,y2;
	printf("start\n");
//	csr.LoadFromMM("matrices/cage3/cage3.mtx");
	csr.LoadFromMM("matrices/ldoor/ldoor.mtx");
//	csr.LoadFromMM("matrices/bbmat/bbmat.mtx");
	x.Create(csr.n);
	x.Fill(1.0f);
	x2.Copy(x);
	y.Create(csr.m);
	y.Fill(0);
	y2.Copy(y);

	ell.TransformFromCSR(csr);
	int res = ell.CopyMatToDevice();
	
//	int res = csr.CopyMatToDevice();
	if(res == -1){
		return 0;
	}
	x.AllocVectorToDevice();
	x.SetVectorValueToDevice();
	y.AllocVectorToDevice();


//	csr.MulLightSpMVOnGPU(x,y);
//	csr.MulOnCPU(x,y);
	ell.MulOnGPU(x,y);
	csr.MklMul(x2,y2);
	double acc1 = 0;
	double acc2 = 0;
	double t1,t2;

	t1 = elasped();
	for(int i = 0; i < TIMES; i++){
		x.SetVectorValueToDevice();
//		csr.MulLightSpMVOnGPU(x,y);
		ell.MulOnGPU(x,y);
		y.GetVectorValueFromDevice();
//		csr.MulOnCPU(x,y);
	}
	t2 = elasped();
	acc1 = t2-t1;
	t1 = elasped();
	for(int i = 0; i < TIMES; i++){
		csr.MklMul(x2,y2);
	}
	t2 = elasped();
	acc2 = t2-t1;


	if(y.Equal(y2,1e-4)){
		printf("same result\n");
	}
	else{
		printf("result wrong\n");
	}
//	y.Dump();
//	y2.Dump();

	printf("elasped\nnaive : %f sec\nMKL : %f sec\n",acc1/TIMES,acc2/TIMES);
	printf("end\n");
	return 0;
}
