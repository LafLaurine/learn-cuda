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
    float e[3];
};


#endif