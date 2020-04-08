#include "matrix.h"

int main(){
	CSR<float> csr;
	ELL<float> ell;
	Vec<float> v;
	printf("start\n");
	csr.LoadFromMM("matrices/cage3/cage3.mtx");
//	csr.LoadFromMM("matrices/ldoor/ldoor.mtx");
	ell.TransformFromCSR(csr);
	ell.Dump();
	printf("end\n");
	return 0;
}
