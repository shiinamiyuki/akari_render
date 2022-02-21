cmake_minimum_required (VERSION 3.12)
project (PBRT-V4-RGB2SPEC-OPT LANGUAGES CXX C)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
	#set(CMAKE_CXX_FLAGS /std:c++17 /MP /arch:AVX2 /WX)
else()
    set(CMAKE_CXX_FLAGS "-lstdc++ -pthread ${CMAKE_CXX_FLAGS}")
endif()

add_executable(rgb2spec_opt rgb2spec_opt.cpp)
