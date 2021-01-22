project(array LANGUAGES CXX CUDA)
add_executable(${PROJECT_NAME} main.cu)
target_include_directories(${PROJECT_NAME} PRIVATE include)
