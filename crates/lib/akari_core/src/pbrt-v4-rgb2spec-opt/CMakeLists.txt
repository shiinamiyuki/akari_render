cmake_minimum_required (VERSION 3.12)
project (PBRT-V4-RGB2SPEC-OPT LANGUAGES CXX C)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rgb2spec_opt STATIC rgb2spec_opt.cpp)