cmake_minimum_required(VERSION 3.15)

project(akari_blender_cpp_ext)
set(CMAKE_CXX_STANDARD 17)
if(WIN32)
    if(MSVC)
        #set(CMAKE_CXX_FLAGS /std:c++17 /MP /arch:AVX2 /WX)
    endif()
else()
    set(CMAKE_CXX_FLAGS "-lstdc++ -pthread ${CMAKE_CXX_FLAGS}")
endif()

add_library(akari_blender_cpp_ext akari_blender_cpp_ext.cpp)
target_include_directories(akari_blender_cpp_ext PUBLIC 
    ${BLENDER_SRC_PATH}/source/blender/
    ${BLENDER_SRC_PATH}/source/blender/makesdna/
    ${BLENDER_SRC_PATH}/source/blender/blenlib/
    ${BLENDER_SRC_PATH}/source/blender/blenkernel/
    ${BLENDER_SRC_PATH}/intern/
    ${BLENDER_SRC_PATH}/intern/guardedalloc
)