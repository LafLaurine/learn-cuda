project(addMatrix LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME} src/chronoCPU.cpp src/chronoGPU.cu src/addMatrix.cu)
target_include_directories(${PROJECT_NAME} PRIVATE include)
