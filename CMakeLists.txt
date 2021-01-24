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


set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/.useless/install)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(RECIPE_EMBREE_ARG "embree")
if(embree_DIR)
    set(RECIPE_EMBREE_ARG "")
endif()
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/recipe.py ${RECIPE_EMBREE_ARG}
               WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_subdirectory(src)

if(NOT COPIED_DLL)
execute_process(COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/useless/copy_dll.py ${CMAKE_SOURCE_DIR}/.useless/install/bin ${CMAKE_BINARY_DIR}/dist
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set(COPIED_DLL TRUE CACHE BOOL "DLL COPIED" FORCE)
endif()


