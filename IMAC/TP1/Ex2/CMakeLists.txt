project(ex2 LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME} src/chronoCPU.cpp src/chronoGPU.cu src/main.cu src/student.cu)
target_include_directories(${PROJECT_NAME} PRIVATE include)
