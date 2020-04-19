#ifndef TEX_H
#define TEX_H

template<typename T>
class TexVec{
public:
	int m;
	void* texObj;
	void* cuArray;
	TexVec(){
		texObj = NULL;
		cuArray = NULL;
	}
	~TexVec(){
		Free();
	}
	void Free();
	void SetTexVec(int m,T* data);
};

#endif
