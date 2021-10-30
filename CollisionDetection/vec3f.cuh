#ifndef VEC3F_H
#define VEC3F_H

#include <ostream>
#include <vector>
#include <assert.h>

#include "mathop.cuh"

#ifndef M_PI
#define M_PI 3.14159f
#endif

class vec3f {
public:
	union {
		struct {
			double x, y, z;
		};
		struct {
			double v[3];
		};
	};

	__device__ __host__ inline vec3f()
	{
		x = 0; y = 0; z = 0;
	}

	__device__ __host__ inline vec3f(const vec3f& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}

	__device__ __host__ inline vec3f(const double* v)
	{
		x = v[0];
		y = v[1];
		z = v[2];
	}

	__device__ __host__ inline vec3f(double x, double y, double z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	__device__ __host__ inline double operator [] (int i) const { return v[i]; }
	__device__ __host__ inline double& operator [] (int i) { return v[i]; }

	__device__ __host__ inline vec3f& operator += (const vec3f& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__device__ __host__ inline vec3f& operator -= (const vec3f& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	__device__ __host__ inline vec3f& operator *= (double t) {
		x *= t;
		y *= t;
		z *= t;
		return *this;
	}

	__device__ __host__ inline vec3f& operator /= (double t) {
		x /= t;
		y /= t;
		z /= t;
		return *this;
	}

	__device__ __host__ inline void negate() {
		x = -x;
		y = -y;
		z = -z;
	}

	__device__ __host__ inline vec3f operator - () const {
		return vec3f(-x, -y, -z);
	}

	__device__ __host__ inline vec3f operator+ (const vec3f& v) const
	{
		return vec3f(x + v.x, y + v.y, z + v.z);
	}

	__device__ __host__ inline vec3f operator- (const vec3f& v) const
	{
		return vec3f(x - v.x, y - v.y, z - v.z);
	}

	__device__ __host__ inline vec3f operator *(double t) const
	{
		return vec3f(x * t, y * t, z * t);
	}

	__device__ __host__ inline vec3f operator /(double t) const
	{
		return vec3f(x / t, y / t, z / t);
	}

	__device__ __host__ inline bool operator ==(vec3f& other) const
	{
		return isEqual(x, other.x) && isEqual(y, other.y) && isEqual(z, other.z);
	}

	// cross product
	__device__ __host__ inline const vec3f cross(const vec3f& vec) const
	{
		return vec3f(y * vec.z - z * vec.y, z * vec.x - x * vec.z, x * vec.y - y * vec.x);
	}

	__device__ __host__ inline double dot(const vec3f& vec) const {
		return x * vec.x + y * vec.y + z * vec.z;
	}

	__device__ __host__ inline void normalize()
	{
		double sum = x * x + y * y + z * z;
		if (sum > GLH_EPSILON_2) {
			double base = double(1.0 / sqrt(sum));
			x *= base;
			y *= base;
			z *= base;
		}
	}

	__device__ __host__ inline double length() const {
		return double(sqrt(x * x + y * y + z * z));
	}

	__device__ __host__ inline vec3f getUnit() const {
		return (*this) / length();
	}

	__device__ __host__ inline bool isUnit() const {
		return isEqual(squareLength(), 1.f);
	}

	//! max(|x|,|y|,|z|)
	__device__ __host__ inline double infinityNorm() const
	{
		return fmax2(fmax2(fabs(x), fabs(y)), fabs(z));
	}

	__device__ __host__ inline vec3f& set_value(const double& vx, const double& vy, const double& vz)
	{
		x = vx; y = vy; z = vz; return *this;
	}

	__device__ __host__ inline bool equal_abs(const vec3f& other) {
		return x == other.x && y == other.y && z == other.z;
	}

	__device__ __host__ inline double squareLength() const {
		return x * x + y * y + z * z;
	}

	__device__ __host__ static vec3f zero() {
		return vec3f(0.f, 0.f, 0.f);
	}

	//! Named constructor: retrieve vector for nth axis
	__device__ __host__ static vec3f axis(int n) {
		assert(n < 3);
		switch (n) {
		case 0: {
			return xAxis();
		}
		case 1: {
			return yAxis();
		}
		case 2: {
			return zAxis();
		}
		}
		return vec3f();
	}

	//! Named constructor: retrieve vector for x axis
	__device__ __host__ static vec3f xAxis() { return vec3f(1.f, 0.f, 0.f); }
	//! Named constructor: retrieve vector for y axis
	__device__ __host__ static vec3f yAxis() { return vec3f(0.f, 1.f, 0.f); }
	//! Named constructor: retrieve vector for z axis
	__device__ __host__ static vec3f zAxis() { return vec3f(0.f, 0.f, 1.f); }

};

__device__ __host__ inline vec3f operator * (double t, const vec3f& v) {
	return vec3f(v.x * t, v.y * t, v.z * t);
}

__device__ __host__ inline vec3f interp(const vec3f& a, const vec3f& b, double t)
{
	return a * (1 - t) + b * t;
}

__device__ __host__ inline vec3f vinterp(const vec3f& a, const vec3f& b, double t)
{
	return a * t + b * (1 - t);
}

__device__ __host__ inline vec3f interp(const vec3f& a, const vec3f& b, const vec3f& c, double u, double v, double w)
{
	return a * u + b * v + c * w;
}

__device__ __host__ inline double clamp(double f, double a, double b)
{
	return fmax2(a, fmin2(f, b));
}

__device__ __host__ inline double vdistance(const vec3f& a, const vec3f& b)
{
	return (a - b).length();
}


__device__ __host__ inline std::ostream& operator<<(std::ostream& os, const vec3f& v) {
	os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
	return os;
}

__device__ __host__ inline void
vmin(vec3f& a, const vec3f& b)
{
	a.set_value(
		fmin2(a[0], b[0]),
		fmin2(a[1], b[1]),
		fmin2(a[2], b[2]));
}

__device__ __host__ inline void
vmax(vec3f& a, const vec3f& b)
{
	a.set_value(
		fmax2(a[0], b[0]),
		fmax2(a[1], b[1]),
		fmax2(a[2], b[2]));
}

//inline vec3f lerp(const vec3f& a, const vec3f& b, float t)
//{
//	return a + t * (b - a);
//}

__device__ __host__ inline int project3(const vec3f& ax,
	const vec3f& p1, const vec3f& p2, const vec3f& p3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);

	double mx1 = fmax3(P1, P2, P3);
	double mn1 = fmin3(P1, P2, P3);

	if (mn1 > 0) return 0;
	if (0 > mx1) return 0;
	return 1;
}

__device__ __host__ inline int project6(vec3f& ax,
	vec3f& p1, vec3f& p2, vec3f& p3,
	vec3f& q1, vec3f& q2, vec3f& q3)
{
	double P1 = ax.dot(p1);
	double P2 = ax.dot(p2);
	double P3 = ax.dot(p3);
	double Q1 = ax.dot(q1);
	double Q2 = ax.dot(q2);
	double Q3 = ax.dot(q3);

	double mx1 = fmax3(P1, P2, P3);
	double mn1 = fmin3(P1, P2, P3);
	double mx2 = fmax3(Q1, Q2, Q3);
	double mn2 = fmin3(Q1, Q2, Q3);

	if (mn1 > mx2) return 0;
	if (mn2 > mx1) return 0;
	return 1;
}

__device__ __host__ inline bool isVec3fEqual(vec3f* a, vec3f* b)
{
	return isEqual(a->x, b->x) && isEqual(a->y, b->y) && isEqual(a->z, b->z);
}

#endif // !VEC3F_H