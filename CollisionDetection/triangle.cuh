//#include <math.h>
#ifndef TRIANGLE_H
#define TRIANGLE_H

typedef struct triangle{
	unsigned int ID;
	unsigned long long int morton;
	double ax, ay, az;
	unsigned int vIdx[3];		// 三角形内只存储vertex的坐标

	__device__ __host__ int selfCheck() {
		for (int i = 0; i < 3; i++) {
			if (vIdx[i] < 0 || vIdx[i] >= 632674) return 1;
		}
		return 0;
	}

	__device__ __host__ int neighborCount(triangle* t) {
		int n11 = (vIdx[0] == t->vIdx[0]);
		int n12 = (vIdx[0] == t->vIdx[1]);
		int n13 = (vIdx[0] == t->vIdx[2]);
		int n21 = (vIdx[1] == t->vIdx[0]);
		int n22 = (vIdx[1] == t->vIdx[1]);
		int n23 = (vIdx[1] == t->vIdx[2]);
		int n31 = (vIdx[2] == t->vIdx[0]);
		int n32 = (vIdx[2] == t->vIdx[1]);
		int n33 = (vIdx[2] == t->vIdx[2]);

		return n11 + n12 + n13 + n21 + n22 + n23 + n31 + n32 + n33;
	}

	__device__ __host__ void set(triangle* target) {
		ID = target->ID;
		for (int i = 0; i < 3; i++) {
			vIdx[i] = target->vIdx[i];
		}
	}

	__device__ __host__ int contact(triangle* other) {
		int nCount = this->neighborCount(other);
		if (nCount > 1) return 0;
		return 1;
	}

} Triangle;

#endif // TRIANGLE_H