project(ex1 LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME} src/main.cpp src/deviceProperties.cu)
