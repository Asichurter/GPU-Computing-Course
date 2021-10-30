#ifndef MATHOP_H
#define MATHOP_H

#include <math.h>

#define     GLH_ZERO                double(0.0)
#define     GLH_EPSILON          double(10e-6)
#define		GLH_EPSILON_2		double(10e-12)
#define     equivalent(a,b)             (((a < b + GLH_EPSILON) &&\
                                                      (a > b - GLH_EPSILON)) ? true : false)

__device__ __host__ inline double lerp(double a, double b, float t)
{
	return a + t * (b - a);
}

__device__ __host__ double fmax2(double a, double b) {
	return (a > b) ? a : b;
}

__device__ __host__ inline double fmin2(double a, double b) {
	return (a < b) ? a : b;
}

__device__ __host__ inline bool isEqual(double a, double b, double tol = GLH_EPSILON)
{
	return fabs(a - b) < tol;
}

__device__ __host__ inline double fmax3(double a, double b, double c)
{
	double t = a;
	if (b > t) t = b;
	if (c > t) t = c;
	return t;
}

__device__ __host__ inline double fmin3(double a, double b, double c)
{
	double t = a;
	if (b < t) t = b;
	if (c < t) t = c;
	return t;
}

#endif // !MATHOP_H
