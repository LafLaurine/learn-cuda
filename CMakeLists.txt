cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(learn-cuda)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures")

# Get projects
add_subdirectory(Personal)
