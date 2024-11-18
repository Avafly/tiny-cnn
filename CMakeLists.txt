cmake_minimum_required(VERSION 3.10)
project(TINY-CNN)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
# set(CMAKE_C_FLAGS "-O3 -DSHOW_RESULTS")
set(CMAKE_C_FLAGS "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "-O3 -DSHOW_RESULTS")
set(CMAKE_CXX_FLAGS "-O3")

set(ENV_ROOT "/your/envs")

# openmp
find_package(OpenMP REQUIRED)
# openblas
set(OPENBLAS_INCLUDE_DIR "${ENV_ROOT}/openblas/include")
set(OPENBLAS_LIB_DIR "${ENV_ROOT}/openblas/lib")

# cnn_struct
add_executable(cnn_struct cnn_struct.c)
target_include_directories(cnn_struct PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_directories(cnn_struct PRIVATE ${OPENBLAS_LIB_DIR})
target_link_libraries(cnn_struct -l:libopenblas.a OpenMP::OpenMP_C -lpthread -lm)
# cnn_const
add_executable(cnn_const cnn_const.c)
target_include_directories(cnn_const PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_link_directories(cnn_const PRIVATE ${OPENBLAS_LIB_DIR})
target_link_libraries(cnn_const -l:libopenblas.a OpenMP::OpenMP_C -lpthread -lm)

# ncnn
set(NCNN_INCLUDE_DIR "${ENV_ROOT}/ncnn/include/ncnn")
set(NCNN_LIBS "${ENV_ROOT}/ncnn/lib/libncnn.a")
# onnxruntime
set(ORT_INCLUDE_DIR "${ENV_ROOT}/onnxruntime/include")
set(ORT_LIBS "${ENV_ROOT}/onnxruntime/lib/libonnxruntime.so")

# ncnn
add_executable(cnn_ncnn cnn_ncnn.cpp)
target_link_libraries(cnn_ncnn ${NCNN_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(cnn_ncnn PRIVATE ${NCNN_INCLUDE_DIR})

# onnxruntime
add_executable(cnn_ort cnn_ort.cpp)
target_link_libraries(cnn_ort ${ORT_LIBS} OpenMP::OpenMP_CXX)
target_include_directories(cnn_ort PRIVATE ${ORT_INCLUDE_DIR})