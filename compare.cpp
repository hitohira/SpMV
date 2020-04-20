#include "matrix.h"
#include "util.h"

const int TIMES = 100;

int measure(const char* filename);

int main(){
	char* mat[] = {
		"matrices/bbmat/bbmat.mtx",
//		"matrices/Chebyshev4/Chebyshev4.mtx",
		"matrices/PR02R/PR02R.mtx",
		"matrices/pwtk/pwtk.mtx",
		"matrices/raefsky3/raefsky3.mtx",
		"matrices/raefsky5/raefsky5.mtx",
		"matrices/rma10/rma10.mtx",
		"matrices/scircuit/scircuit.mtx",
		"matrices/TSOPF_RS_b300_c3/TSOPF_RS_b300_c3.mtx",
		"matrices/twotone/twotone.mtx",
	};
	int numMat = sizeof(mat)/sizeof(char*);
	for(int i = 0; i < numMat; i++){
		measure(mat[i]);
	}
	return 0;
}


int measure(const char* fileName){
	CSR<float> csr;
	ELL<float> ell;
	COO<float> coo;
	Vec<float> x;
	Vec<float> y;
	printf("%s\n",fileName);
	csr.LoadFromMM(fileName);
	if(csr.m == 0){
		printf("load error %s\n",fileName);
		return -1;	
	}

	x.Create(csr.n);
	x.Fill(1.0f);
	y.Create(csr.m);
	y.Fill(0);

	csr.CopyMatToDevice();
	ell.TransformFromCSR(csr);
	ell.CopyMatToDevice();
	coo.TransformFromCSR(csr);
	coo.CopyMatToDevice();
	x.AllocVectorToDevice();
	x.SetVectorValueToDevice();
	x.SetTexVec();
	y.AllocVectorToDevice();


	double t1,t2;
	
	int numFmt = 6;
	char* fmt[] = {
		"coo_mkl", "csr_mkl", "csr_cusparse", "csr_lightspmv", "ell_naive", "ell_tex",
	};
	for(int xx = 0; xx < numFmt; xx++){
		t1 = elasped();
		for(int i = 0; i < TIMES; i++){
			switch(xx){
				case 0:
				coo.MulOnCPU(x,y);
				break;
				case 1:
				csr.MklMul(x,y);
				break;
				case 2:
			x.SetVectorValueToDevice();
				csr.CuSparseMul(x,y);
			y.GetVectorValueFromDevice();
				break;
				case 3:
			x.SetVectorValueToDevice();
				csr.MulLightSpMVOnGPU(x,y);
			y.GetVectorValueFromDevice();
				break;
				case 4:
			x.SetVectorValueToDevice();
				ell.MulOnGPU(x,y);
			y.GetVectorValueFromDevice();
				break;
				case 5:
			x.SetVectorValueToDevice();
			x.SetTexVec();
				ell.MulOnGPUWithTex(x,y);
			y.GetVectorValueFromDevice();
				break;
				default: break;
			}
		}
		t2 = elasped();
		printf("%s ave = %f ms\n",fmt[xx],(t2-t1)*1e3 / TIMES);
	}
	return 0;
}
