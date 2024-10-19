cmake_minimum_required(VERSION 3.29...3.31)
set(VCPKG_BUILD_TYPE release)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake)
project(akari-cpp-ext)
# if on windows and not using clang-cl and msvc

set(CMAKE_CXX_STANDARD 23)

message(STATUS "${CMAKE_CXX_COMPILER}")
add_subdirectory(ext)

find_package(OpenImageIO CONFIG REQUIRED)
set(embree_ROOT ${CMAKE_SOURCE_DIR}/../cmake-build-embree/install)
find_package(embree 4.3 REQUIRED)
add_library(akari_cpp_ext SHARED src/rt.cpp src/image.cpp include/rust-api.h)

target_include_directories(akari_cpp_ext PUBLIC include)
target_link_libraries(akari_cpp_ext embree embree_sse42 embree_avx embree_avx2  OpenImageIO::OpenImageIO)

install(TARGETS akari_cpp_ext
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        )
