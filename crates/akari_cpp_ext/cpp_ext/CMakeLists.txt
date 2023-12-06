cmake_minimum_required(VERSION 3.20)

project(akari_cpp_ext)
set(CMAKE_CXX_STANDARD 20)
file(READ "../../../blender_src_path.txt" BLENDER_SRC_PATH)
string(STRIP ${BLENDER_SRC_PATH} BLENDER_SRC_PATH)
if(WIN32)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "/arch:AVX2 /WX /GR- /D_HAS_EXCEPTIONS=0 /EHsc ${CMAKE_CXX_FLAGS}")
    endif()
else()
    set(CMAKE_CXX_FLAGS "-lstdc++ -pthread -mavx2 -fno-rtti -fno-exceptions ${CMAKE_CXX_FLAGS}")
endif()

add_library(akari_cpp_ext akari_cpp_ext.cpp rgb2spec_opt.cpp)
target_include_directories(akari_cpp_ext PUBLIC
    ${BLENDER_SRC_PATH}/source/blender/
    ${BLENDER_SRC_PATH}/source/blender/makesdna/
    ${BLENDER_SRC_PATH}/source/blender/blenlib/
    ${BLENDER_SRC_PATH}/source/blender/blenkernel/
    ${BLENDER_SRC_PATH}/intern/
    ${BLENDER_SRC_PATH}/intern/guardedalloc
)