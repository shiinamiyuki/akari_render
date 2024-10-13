cmake_minimum_required(VERSION 3.26)
project(akari-cpp-ext)
add_subdirectory(ext)

add_library(akari_cpp_ext STATIC ext-impl.cpp rust-api.h)
target_link_libraries(akari_cpp_ext embree embree_sse42 embree_avx embree_avx2)