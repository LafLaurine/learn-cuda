#ifndef RAYH
#define RAYH
#include "vector3.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vector3& a, const vector3& b) { A = a; B = b; }
        __device__ vector3 origin() const       { return A; }
        __device__ vector3 direction() const    { return B; }
        __device__ vector3 point_at_parameter(float t) const { return A + t*B; }
        vector3 A;
        vector3 B;
};

#endif