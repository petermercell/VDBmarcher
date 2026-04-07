cmake_minimum_required(VERSION 3.11)
project(VDBRender LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED NUKE_VERSION)
    set(NUKE_VERSION "17.0v1")
endif()
set(NDKDIR /opt/Nuke${NUKE_VERSION})
if(NOT EXISTS ${NDKDIR}/libDDImage.so)
    message(FATAL_ERROR "Nuke not found: ${NDKDIR}")
endif()
include_directories(${NDKDIR}/include)

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_compile_options(-fPIC)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

set(VDB_SOURCES
    src/VDBRenderIop.h
    src/VDBRenderIop.cpp
    src/VDBValidate.cpp
    src/VDBLighting.cpp
    src/VDBEngine.cpp
    src/VDBRenderCUDA.cpp
    src/VDBGridLoader.cpp
    src/VDBViewport.cpp
    src/VDBCUDARenderer.h
    src/VDBCUDARenderer.cu
    src/CVDBLoader.h
    src/VDBRenderConfig.h
    src/VDBEnvironment.h
    src/VDBEnvironment.cpp
    # ── Deep output ──
    src/VDBDeepEngine.cpp
)

add_library(VDBRender SHARED ${VDB_SOURCES})
target_include_directories(VDBRender PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(VDBRender PRIVATE
    ${NDKDIR}/libDDImage.so
    CUDA::cudart
    OpenGL::GL
    Threads::Threads
)
set_target_properties(VDBRender PROPERTIES
    CUDA_ARCHITECTURES "75;86;89"
    CUDA_SEPARABLE_COMPILATION ON
)
# Report register usage per kernel at build time (invaluable for occupancy tuning)
target_compile_options(VDBRender PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)

install(TARGETS VDBRender DESTINATION ${CMAKE_INSTALL_PREFIX})
message(STATUS "VDBRender: CVDB codebook volume renderer (CUDA)")
message(STATUS "  Dependencies: CUDA + Nuke DDImage (no OpenVDB)")
message(STATUS "  Nodes: VDBRender (flat Iop) + VDBDeepRead (DeepFilterOp)")
