project(imac-tp3 LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME} src/main.cu src/student.cu src/chronoCPU.cpp src/chronoGPU.cu)
target_include_directories(${PROJECT_NAME} PRIVATE include)
