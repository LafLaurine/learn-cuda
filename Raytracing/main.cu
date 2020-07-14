#include <iostream>
#include "vector3.h"
#include "ray.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// device function can only be called from other device or global functions (can't be called from host)

__device__ bool hit_sphere(const vector3& center, float radius, const ray& r) {
    vector3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc,oc) - radius * radius;
    float discriminant = b*b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ vector3 color(const ray& r) {
    if (hit_sphere(vector3(0,0,-1), 0.5, r))
        return vector3(1,0,0);
    vector3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vector3(1.0, 1.0, 1.0) + t*vector3(0.5, 0.7, 1.0);
}

__global__ void render(vector3 *fb, int max_x, int max_y, vector3 lower_left_corner, vector3 horizontal, vector3 vertical, vector3 origin) {
    // we identify the coordinates of each thread in the image (i,j)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    /*int pixel_index = j*max_x*3 + i*3;
    fb[pixel_index + 0] = float(i) / max_x;
    fb[pixel_index + 1] = float(j) / max_y;
    fb[pixel_index + 2] = 0.2;*/
    int pixel_index = j*max_x + i;
    //fb[pixel_index] = vector3( float(i) / max_x, float(j) / max_y, 0.2f);
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin,lower_left_corner + u*horizontal + v*vertical);
    fb[pixel_index] = color(r);
}

int main(void) {    
    int nx = 1200;
    int ny = 600;
    // blocks of 8x8 threads
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // allocate an nx*ny image-sized frame buffer on the host to hold the RGB float values calculated by the GPU
    int num_pixels = nx*ny;
    size_t fb_size = 3*num_pixels*sizeof(vector3);

    // allocate FB
    vector3 *fb;
    // cudaMallocManaged allocates unified memory
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render<<<blocks, threads>>>(fb, nx, ny, vector3(-2.0, -1.0, -1.0),vector3(4.0, 0.0, 0.0),vector3(0.0, 2.0, 0.0),vector3(0.0, 0.0, 0.0));
    checkCudaErrors(cudaGetLastError());
    // cudaDeviceSynchronize lets the CPU know when the GPU is done rendering
    checkCudaErrors(cudaDeviceSynchronize());

    // output frame buffer as image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            /*size_t pixel_index = j*3*nx + i*3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99*r);
            int ig = int(255.99*g);
            int ib = int(255.99*b);
            std::cout << ir << " " << ig << " " << ib << "\n";*/

            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(fb));
}