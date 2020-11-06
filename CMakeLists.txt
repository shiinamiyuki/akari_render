cmake_minimum_required(VERSION 3.12)
project(AkariRender)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_CXX_STANDARD 17)
option(AKR_BUILD_GUI "Build Gui / Editor" OFF)
option(AKR_NO_GPU "Turn off GPU support" OFF)
option(AKR_ENABLE_EMBREE "Use Embree as ray intersection backend" OFF)
option(AKR_USE_PREBUILD_EMBREE "Use prebuilt binary" OFF)
option(AKR_ENABLE_ASL "Enable Akari Shading Language for procedural shading" OFF)
option(AKR_ENABLE_PYTHON "Build Python Binding" OFF)
option(AKR_ENABLE_OIDN "Use Intel OpenImageDenoise" OFF)
option(AKR_USE_PREBUILD_OIDN "Use prebuilt Intel OpenImageDenoise" OFF)
option(AKR_ENABLE_OPENEXR "Use OpenEXR for storing HDR images" OFF)
option(AKR_ENABLE_NETWORK "Enable Network Rendering" OFF)
option(AKR_ENABLE_ASSIMP "Use assimp to import more assets" OFF)

include_directories(${CMAKE_SOURCE_DIR}/external/glm)
include_directories(${CMAKE_SOURCE_DIR}/external/misc)
include_directories(${CMAKE_SOURCE_DIR}/external/json)
include_directories(${CMAKE_SOURCE_DIR}/external/cxxopts/include)
include_directories(${CMAKE_SOURCE_DIR}/external/magic_enum/include)

if(EXISTS ${CMAKE_BINARY_DIR}/dist)
else()
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dist)
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dist/plugins)
endif()
function(copy_to_dist target)
    add_custom_command(TARGET ${target} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target}>  ${CMAKE_BINARY_DIR}/dist/)
endfunction()


set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "")
set(CMAKE_STATIC_LIBRARY_PREFIX_CXX "")
set(AKR_CORE_EXT_LIBS "")
set(AKR_EXT_INCLUDES "")
set(AKR_COMPILE_DEFINITIONS AKR_ENABLE_CPU)
if(MSVC)
    set(AKR_CXX_FLAGS /std:c++17 /MP /arch:AVX2 /WX)
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ")
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} ")
else()
    set(CMAKE_CXX_FLAGS -fPIC)
    set(AKR_CXX_FLAGS -mavx2)
    set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} stdc++fs)
endif()

if(NOT WIN32)
    set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} pthread dl)
    set(AKR_COMPILE_DEFINITIONS ${AKR_COMPILE_DEFINITIONS} AKR_PLATFORM_LINUX)
else()
    set(AKR_COMPILE_DEFINITIONS ${AKR_COMPILE_DEFINITIONS} AKR_PLATFORM_WINDOWS)
    add_compile_definitions(_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()


find_package (ZLIB REQUIRED) # cpp no good
if(AKR_ENABLE_ASSIMP)
# assimp attempts to build its own zlib, no good
    set(AKR_EXT_INCLUDES ${AKR_EXT_INCLUDES} ${CMAKE_SOURCE_DIR}/external/assimp/include)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/assimp)
endif()




# if (NOT ZLIB_FOUND)
#   # Build zlib
#     set (ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
#     set (ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
#     add_subdirectory (external/zlib)

#     set (ZLIB_LIBRARIES zlib)
#     set (ZLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/external/zlib ${CMAKE_CURRENT_BINARY_DIR}/external/zlib)

#     # try to make openexr happy about this...
#     set (ZLIB_LIBRARY zlib)
#     set (ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/zlib ${CMAKE_CURRENT_BINARY_DIR}/external/zlib)
#     set (ZLIB_FOUND TRUE)
#     set(ASSIMP_BUILD_ZLIB OFF CACHE BOOL " " FORCE)
#     include_directories(${ZLIB_INCLUDE_DIRS})  # yuck, but so openexr/ptex can find zlib.h...
# else()
#     set(ASSIMP_BUILD_ZLIB OFF CACHE BOOL " " FORCE)
# endif ()
set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} ${ZLIB_LIBRARIES})


if(AKR_ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    add_subdirectory(external/pybind11)
    add_compile_definitions(AKR_ENABLE_PYTHON)
endif()
if(AKR_ENABLE_OPENEXR)
    set(ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
    set(OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
    set(OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
    set(ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
    set(PYILMBASE_ENABLE             OFF CACHE BOOL " " FORCE)
    set(OPENEXR_BUILD_UTILS          OFF CACHE BOOL " " FORCE)
    set(BUILD_TESTING                OFF CACHE BOOL " " FORCE)
    set(INSTALL_OPENEXR_EXAMPLES     OFF CACHE BOOL " " FORCE)
    set(AKR_EXT_INCLUDES ${AKR_EXT_INCLUDES} external/openexr/OpenEXR/IlmImf)
    add_subdirectory(external/openexr)
endif()

set(AKR_EXT_INCLUDES ${AKR_EXT_INCLUDES} external/tinyobjloader)

set(AKR_EXT_INCLUDES ${AKR_EXT_INCLUDES} external/fmt-6.1.2/include)
add_subdirectory(external/fmt-6.1.2)
set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} fmt)

macro(add_build_util target )
    add_executable(${target} ${ARGN})
    set(${target}_SOURCE ${ARGN})
endmacro()

include(cmake/Embree.cmake)
include(cmake/OIDN.cmake)
include(cmake/CUDA.cmake)


file(GLOB PYBIND11_HEADERS external/pybind11/include/pybind11.h external/pybind11/include/stl.h)
file(GLOB AKR_CORE_SRC src/akari/core/*.*)
file(GLOB AKR_PYBIND_SRC src/akari/core/nodes/*.*)



add_library(libakari_core SHARED ${AKR_CORE_SRC})
target_link_libraries(libakari_core PUBLIC  ${AKR_CORE_EXT_LIBS} ${AKR_CUDA_LIBS})
set_target_CUDA_props(libakari_core)
if(AKR_ENABLE_PYTHON)
    target_link_libraries(libakari_core ${Python_LIBRARIES})
endif()
target_compile_options(libakari_core PUBLIC ${AKR_CXX_FLAGS})
target_compile_definitions(libakari_core PUBLIC ${AKR_COMPILE_DEFINITIONS})
target_include_directories(libakari_core PUBLIC src/ ${AKR_EXT_INCLUDES})
target_include_directories(libakari_core PUBLIC external/pybind11/include)
target_include_directories(libakari_core PUBLIC ${Python_INCLUDE_DIRS} ${EMBREE_INCLUDE_DIRS})
add_library(akari::core ALIAS libakari_core)
copy_to_dist(libakari_core)
if(AKR_ENABLE_PYTHON)
    target_include_directories(libakari_core ${Python_INCLUDE_DIRS})
endif()
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL  "3.16")
if(NOT WIN32)
    target_precompile_headers(libakari_core
        PUBLIC ${PYBIND11_HEADERS}
        <vector>
        <string>
        <unordered_map>
        <memory>
        <thread>
        <atomic>)
endif()
endif()


file(GLOB AKR_RENDER_SRC src/akari/render/*.*)
if(AKR_ENABLE_CUDA)
    file(GLOB AKR_RENDER_CUDA_SRC src/akari/render/cuda/*.*)
    cuda_compile_and_embed (AKR_EMBEDDED_PTX src/akari/render/cuda/optix.cu optix.cu)
    add_library (libakari_embedded_ptx STATIC
      ${AKR_EMBEDDED_PTX}
    )
endif()
add_library(libakari_render SHARED ${AKR_RENDER_SRC} ${AKR_RENDER_CUDA_SRC})
target_include_directories(libakari_render PUBLIC src/)
target_link_libraries(libakari_render PUBLIC akari::core)
copy_to_dist(libakari_render)
set_target_CUDA_props(libakari_render)
set_target_properties(libakari_render PROPERTIES UNITY_BUILD ON)
add_library(akari::render ALIAS libakari_render)
if(AKR_ENABLE_CUDA)
    target_link_libraries(libakari_render PUBLIC libakari_embedded_ptx)
endif()

function(add_plugin target)
    add_library(${target} SHARED ${ARGN})
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${target} PROPERTIES PREFIX "")
    # set_target_properties(${target} PROPERTIES
    #     LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
    #     ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
    #     RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/plugins/")
    target_link_libraries(${target} akari::render)
    add_custom_command(TARGET ${target} POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${target}>  ${CMAKE_BINARY_DIR}/dist/plugins/)
    set_target_CUDA_props(${target})
endfunction()

add_subdirectory(src/akari/render)
add_subdirectory(src/akari/plugins)

add_executable(akari src/akari/cmd/akari.cpp )
target_include_directories(akari PUBLIC src/)
target_link_libraries(akari libakari_render)
set_target_CUDA_props(akari)
copy_to_dist(akari)

add_executable(akari-import src/akari/cmd/akari-import.cpp )
target_include_directories(akari-import PUBLIC src/)
copy_to_dist(akari-import)
set_target_CUDA_props(akari-import)
target_link_libraries(akari-import akari::core)
if(AKR_ENABLE_ASSIMP)
target_link_libraries(akari-import assimp)
    target_compile_definitions(akari-import PUBLIC AKR_ENABLE_ASSIMP)
endif()

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
file(GLOB AKR_TEST src/akari/tests/*.*)
add_executable(akari-test ${AKR_TEST} src/akari/cmd/akari-test.cpp)
target_link_libraries(akari-test gtest libakari_core)
add_test(NAME akari COMMAND akari-test)
copy_to_dist(akari-test)
set_target_CUDA_props(akari-test)

