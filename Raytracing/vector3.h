#ifndef VECTOR3H
#define VECTOR3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

// we need to precise __host__ __device__ for executing members both on CPU and GPU

class vector3  {
public:
    __host__ __device__ vector3() {}
    __host__ __device__ vector3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; };
    __host__ __device__ inline float x() const { return e[0]; };
    __host__ __device__ inline float y() const { return e[1]; };
    __host__ __device__ inline float z() const { return e[2]; };
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }
    __host__ __device__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline vector3& operator/=(const vector3 &v2);
    __host__ __device__ inline vector3& operator/=(const float t);
    __host__ __device__ inline vector3& operator*=(const vector3 &v2);
    __host__ __device__ inline vector3& operator*=(const float t);

    float e[3];
};

__host__ __device__ inline vector3 operator+(const vector3 &v1, const vector3 &v2) {
    return vector3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vector3 operator/(vector3 v, float t) {
    return vector3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vector3 operator*(const vector3 &v1, const vector3 &v2) {
    return vector3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}


__host__ __device__ inline vector3 operator*(float t, const vector3 &v) {
    return vector3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vector3& vector3::operator/=(const vector3 &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

__host__ __device__ inline vector3& vector3::operator/=(const float t) {
    float k = 1.0/t;
    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline vector3 unit_vector(vector3 v) {
    return v / v.length();
}


#endif