cmake_minimum_required(VERSION 3.12)
project(AkariRender)

set(CMAKE_CXX_STANDARD 17)
option(AKR_BUILD_GUI "Build Gui / Editor" OFF)
option(AKR_PYTHON_SUPPORT "Enable Python Support" OFF)


include_directories(${CMAKE_SOURCE_DIR}/external/glm)
include_directories(${CMAKE_SOURCE_DIR}/external/misc)
include_directories(${CMAKE_SOURCE_DIR}/external/json)
include_directories(${CMAKE_SOURCE_DIR}/external/cxxopts/include)
include_directories(${CMAKE_SOURCE_DIR}/external/magic_enum/include)

# include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/include)

set(AKR_CORE_EXT_LIBS "")
if(MSVC)
    set(AKR_CXX_FLAGS /std:c++17 /MD  /MP /GL /arch:AVX2 /WX)
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /LTCG:INCREMENTAL")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} /LTCG:INCREMENTAL")
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} /LTCG:INCREMENTAL")
else()
    set(CMAKE_CXX_FLAGS -fPIC)
    set(AKR_CXX_FLAGS -mavx2)
    set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} stdc++fs)
endif()

if(NOT WIN32)
    set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} pthread dl)
endif()


#
#if(AKR_BUILD_GUI)
#    include_directories(external/glfw-3.3.2/include)
#    add_subdirectory(external/glfw-3.3.2)
#    set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} glfw)
#endif()
#

include_directories(external/tinyobjloader)

add_library(tinyobjloader external/tinyobjloader/tiny_obj_loader.cc)
target_include_directories(tinyobjloader PRIVATE external/tinyobjloader/)
set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} tinyobjloader)


include_directories(external/fmt-6.1.2/include)
add_subdirectory(external/fmt-6.1.2)
set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} fmt)

# include_directories(external/asmjit/src)
# add_subdirectory(external/asmjit)


# include(cmake/GUI.cmake)
# include(cmake/Embree.cmake)
# include(cmake/OIDN.cmake)
# include(cmake/Vulkan.cmake)
# include(cmake/LLVM.cmake)

##set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} embree embree_sse42 embree_avx embree_avx2 lexers math simd tasking sys)
#

file(GLOB AKR_COMMON_SRC src/akari/common/*.*)
file(GLOB AKR_CORE_SRC src/akari/core/*.* )
add_library(akari_core ${AKR_CORE_SRC})
target_link_libraries(akari_core  ${AKR_CORE_EXT_LIBS})
target_compile_options(akari_core PUBLIC ${AKR_CXX_FLAGS})
target_include_directories(akari_core PUBLIC src/)

file(GLOB AKR_KERNEL_SRC src/akari/kernel/*.* src/akari/kernel/*/*.*)


add_executable(akari src/akari/app/akari.cpp)
target_include_directories(akari PUBLIC src/)
# target_link_libraries(akari akari_core)