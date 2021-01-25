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

option(AKR_BACKEND_EMBREE "Enable cpu backend with embree" ON)
option(AKR_BACKEND_CUDA "Enable CUDA + Optix7 backend" OFF)
option(AKR_BUILD_GUI "Build GUI" OFF)
option(AKR_ENABLE_ENGINE "Build AkariEngine" OFF)
option(AKR_USE_OPENVDB "Use OpenVDB" ON)
option(AKR_BUILD_DIFF "Build diff-akari" OFF)

set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/.useless/install)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(RECIPE_ARGS "")
if(NOT embree_DIR)
    set(RECIPE_ARGS ${RECIPE_ARGS} embree)
endif()
if(AKR_USE_OPENVDB)
    set(RECIPE_ARGS ${RECIPE_ARGS} openvdb)
endif()
if(AKR_BUILD_GUI)
    set(RECIPE_ARGS ${RECIPE_ARGS} gui)
endif()
if(AKR_BUILD_DIFF)
    set(RECIPE_ARGS ${RECIPE_ARGS} diff)
endif()
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/recipe.py ${RECIPE_ARGS}
               WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_subdirectory(src)

if(NOT COPIED_DLL)
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/useless/copy_dll.py ${CMAKE_SOURCE_DIR}/.useless/install/bin ${CMAKE_BINARY_DIR}/dist
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set(COPIED_DLL TRUE CACHE BOOL "DLL COPIED" FORCE)
endif()


