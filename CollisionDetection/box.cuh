#ifndef BOX_H
#define BOX_H

#include "triangle.cuh"
#include "vec3f.cuh"
#include "mathop.cuh"

typedef struct box {
	double x1, x2, y1, y2, z1, z2;
	int count;
	int init;
	
	__device__ __host__ void set(Triangle* t, vec3f* vs) {
		vec3f *v1 = &vs[t->vIdx[0]], *v2 = &vs[t->vIdx[1]], *v3 = &vs[t->vIdx[2]];
		x1 = fmin3(v1->x, v2->x, v3->x);
		x2 = fmax3(v1->x, v2->x, v3->x);
		y1 = fmin3(v1->y, v2->y, v3->y);
		y2 = fmax3(v1->y, v2->y, v3->y);
		z1 = fmin3(v1->z, v2->z, v3->z);
		z2 = fmax3(v1->z, v2->z, v3->z);
		init = 1;
	}

	__device__ __host__ void merge(box* b1, box* b2) {
		x1 = fmin2(b1->x1, b2->x1);
		x2 = fmax2(b1->x2, b2->x2);
		y1 = fmin2(b1->y1, b2->y1);
		y2 = fmax2(b1->y2, b2->y2);
		z1 = fmin2(b1->z1, b2->z1);
		z2 = fmax2(b1->z2, b2->z2);
		init = 1;
	}

	__device__ __host__ int overlap(box* other) {
		//if ((x1 < other->x2) && (other->x1 < x2)) return 1;
		//if (((x1 - other->x2) * (other->x1 - x2)) > 0) return 1;
		//if (((y1 - other->y2) * (other->y1 - y2)) > 0) return 1; 
		//if (((z1 - other->z2) * (other->z1 - z2)) > 0) return 1;
		if ((x1 - other->x2) * (other->x1 - x2) > 0 || ((y1 - other->y2) * (other->y1 - y2)) > 0 || ((z1 - other->z2) * (other->z1 - z2)) > 0) return 1;
		return 0;
	}

	__device__ __host__ int overlapTest(box* other) {
		if (other == NULL) return 2;

		if (x1 - other->x2 < 0) return 1;
		else return 0;
	}

	__device__ __host__ int selfCheck() {
		return init;
	}
} Box;


__device__ __host__ int checkBoxOverlap(Box* a, Box* b) {
	if ((a->x1 - b->x2) * (b->x1 - a->x2) > 0 && (a->y1 - b->y2) * (b->y1 - a->y2) > 0 && (a->z1 - b->z2) * (b->z1 - a->z2) > 0) return 1;
	return 0;
}

#endif // !BOX_H
