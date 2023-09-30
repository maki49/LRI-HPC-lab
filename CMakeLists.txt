cmake_minimum_required(VERSION 3.16)
project(lri_gpu)

option(ENALE_MPI "Enable MPI" ON)
option(ENABLE_CUDA "Enable CUDA" ON)

aux_source_directory(src ALL_SRCS)
add_executable(${PROJECT_NAME} ${ALL_SRCS})

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