cmake_minimum_required(VERSION 3.16)
project(lri_gpu)

option(ENALE_MPI "Enable MPI" ON)
option(ENABLE_CUDA "Enable CUDA" ON)

aux_source_directory(src ALL_SRCS)
add_executable(${PROJECT_NAME} ${ALL_SRCS})
include_directories(LibRI/include LibComm/include)

#openmp
  find_package(OpenMP REQUIRED)
  target_link_libraries(${ABACUS_BIN_NAME} OpenMP::OpenMP_CXX)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  add_link_options(${OpenMP_CXX_LIBRARIES})

#blas
find_package(BLAS REQUIRED)
target_link_libraries(${PROJECT_NAME} ${BLAS_LIBRARIES})

if(ENALE_MPI)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
    # add_compile_definitions(__MPI)  # macros in source code
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES
    60 # P100
    70 # V100
    75 # T4
    80 # A100
  )
    target_link_libraries(${PROJECT_NAME} cublas cusparse)
endif()

# if(ENABLE_OPENACC)
#     set(CMAKE_CXX_COMPILER pgc++)
#     set(OpenACC_C_FLAGS "-acc")
#     set(OpenACC_CXX_FLAGS "-acc")
#     find_package(OpenACC REQUIRED)
#     target_link_libraries(${PROJECT_NAME}  OpenACC::OpenACC_CXX)
#     add_link_options(${OpenACC_CXX_LIBRARIES})
# endif()