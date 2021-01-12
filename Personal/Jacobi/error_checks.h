#ifndef ERRORCHECKS
#define ERRORCHECKS

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(errarg) __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrorMsgFunc(errstr, __FILE__, __LINE__)

inline void __checkErrorFunc(cudaError_t errarg, const char* file, const int line) {
    if(errarg) {
        fprintf(stderr, "Error at %s(%i)\n", file, line);
        exit(EXIT_FAILURE)
    }
}

inline void __checkErrorMsgFunc(const char* errstr, const char* file, const int line) {
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "Error: %s at %s(%i): %s\n", errstr, file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE)
    }
}

#endif