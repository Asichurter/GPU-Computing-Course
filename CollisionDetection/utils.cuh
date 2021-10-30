#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>

#include "box.cuh"

__host__ void printBox(Box* b, const char* title)
{
	printf("$ %s: [(%f, %f)->(%f), (%f, %f)->(%f), (%f, %f)->(%f)]\n", title, b->x1, b->x2, (b->x1 + b->x2)/2, b->y1, b->y2, (b->y1 + b->y2) / 2, b->z1, b->z2, (b->z1 + b->z2) / 2);
}

#endif // !UTILS_H