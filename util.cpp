#include "util.h"
#include <stdio.h>
#include <sys/time.h>

double elasped(){
	struct timeval tv;
	gettimeofday(&tv,NULL);
	return tv.tv_sec + tv.tv_usec*1e-6;
}

