project(imac-tp2 LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME} src/main.cu src/student.cu src/lodepng.cpp src/conv_utils.cpp src/chronoCPU.cpp src/chronoGPU.cu)
target_include_directories(${PROJECT_NAME} PRIVATE include)
