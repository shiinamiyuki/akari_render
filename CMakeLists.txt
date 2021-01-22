cmake_minimum_required(VERSION 3.12)
# if(NOT CMAKE_TOOLCHAIN_FILE)
#   set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake
#     CACHE STRING "Vcpkg toolchain file")
# endif()
project(AkariRender)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "")
set(CMAKE_STATIC_LIBRARY_PREFIX_CXX "")


set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/useless/.useless/install)

add_subdirectory(src)


