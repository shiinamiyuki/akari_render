cmake_minimum_required(VERSION 3.12)
project(AkariRender)

set(CMAKE_CXX_STANDARD 17)
option(AKR_BUILD_GUI "Build Gui / Editor" OFF)
option(AKR_PYTHON_SUPPORT "Enable Python Support" OFF)
option(AKR_ENABLE_EMBREE "Use Embree as ray intersection backend" OFF)
option(AKR_USE_PREBUILD_EMBREE "Use prebuilt binary" OFF)
option(AKR_ENABLE_ASL "Enable Akari Shading Language for procedural shading" OFF)
option(AKR_ENABLE_PYTHON "Build Python Binding" OFF)

include_directories(${CMAKE_SOURCE_DIR}/external/glm)
include_directories(${CMAKE_SOURCE_DIR}/external/misc)
include_directories(${CMAKE_SOURCE_DIR}/external/json)
include_directories(${CMAKE_SOURCE_DIR}/external/cxxopts/include)
include_directories(${CMAKE_SOURCE_DIR}/external/magic_enum/include)


set(AKR_CORE_EXT_LIBS "")
set(AKR_COMPILE_DEFINITIONS AKR_ENABLE_CPU)
if(MSVC)
    set(AKR_CXX_FLAGS /std:c++17 /MD  /MP /GL /arch:AVX2 /WX)
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /LTCG:INCREMENTAL")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} /LTCG:INCREMENTAL")
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} /LTCG:INCREMENTAL")
else()
    set(CMAKE_CXX_FLAGS -fPIC)
    set(AKR_CXX_FLAGS -mavx2 )
    set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} stdc++fs)
endif()

if(NOT WIN32)
    set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} pthread dl)
    set(AKR_COMPILE_DEFINITIONS ${AKR_COMPILE_DEFINITIONS} AKR_PLATFORM_LINUX)
else()
    set(AKR_COMPILE_DEFINITIONS ${AKR_COMPILE_DEFINITIONS} AKR_PLATFORM_WINDOWS)
    add_compile_definitions(_ENABLE_EXTENDED_ALIGNED_STORAGE)
endif()

if(AKR_ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    add_subdirectory(external/pybind11)
    add_compile_definitions(AKR_ENABLE_PYTHON)
endif()
include_directories(external/tinyobjloader)

include_directories(external/fmt-6.1.2/include)
add_subdirectory(external/fmt-6.1.2)
set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} fmt)

include_directories(${CMAKE_SOURCE_DIR}/external/pugixml/src)
add_subdirectory(external/pugixml)
set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} pugixml)

# ---------------- Akari Unified Shading Language Compiler
file(GLOB ASL_COMPILER_SRC src/akaric/*.*)
add_executable(akaric ${ASL_COMPILER_SRC})
target_include_directories(akaric PUBLIC src/)
target_link_libraries(akaric fmt)


# ---------------- Begin Build Utils --------------------
add_executable(akari-configure src/akari/tools/configure.cpp)
target_link_libraries(akari-configure fmt)
set_target_properties(akari-configure 
        PROPERTIES  CUDA_ARCHITECTURES OFF)

add_executable(akari-soac src/akari/tools/soac.cpp)
target_link_libraries(akari-soac fmt)
set_target_properties(akari-soac
        PROPERTIES  CUDA_ARCHITECTURES OFF)
macro(add_soa_header target source)
    add_custom_command(OUTPUT ${target}
        COMMAND akari-soac ${source} ${target}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS ${source} akari-soac ${CMAKE_SOURCE_DIR}/src/akari/common/soa.h)
endmacro()

# ---------------- End Build Utils --------------------


include(cmake/Embree.cmake)
include(cmake/CUDA.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/akari.conf)
else()
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources/akari.conf DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(AKR_CONFIG_H ${CMAKE_CURRENT_SOURCE_DIR}/src/akari/common/config.h )
add_custom_command(OUTPUT ${AKR_CONFIG_H}
    COMMAND akari-configure ${CMAKE_CURRENT_BINARY_DIR}/akari.conf ${AKR_CONFIG_H}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/akari.conf akari-configure)


file(GLOB PYBIND11_HEADERS external/pybind11/include/pybind11.h external/pybind11/include/stl.h)
file(GLOB AKR_COMMON_SRC src/akari/common/*.*)
file(GLOB AKR_SHADING_SRC src/akari/shading/*.*)
file(GLOB AKR_CORE_SRC src/akari/core/*.*)
file(GLOB AKR_PYBIND_SRC src/akari/core/nodes/*.*)

file(GLOB AKR_KERNEL_COMMON_SRC src/akari/kernel/*.*)
file(GLOB AKR_KERNEL_CPU_SRC src/akari/kernel/integrators/cpu/*.*)
file(GLOB AKR_KERNEL_GPU_SRC ${AKR_KERNEL_SOA_H} src/akari/kernel/integrators/gpu/*.* src/akari/kernel/cuda/*.*)


if(AKR_ENABLE_CUDA)
    cmake_policy(SET CMP0104 NEW)
    file(GLOB AKR_KERNEL_GPU_CUDA_SRC src/akari/kernel/integrators/gpu/cuda/*.*)
    set(AKR_KERNEL_SRC ${AKR_KERNEL_COMMON_SRC} ${AKR_KERNEL_CPU_SRC} ${AKR_KERNEL_GPU_SRC} ${AKR_KERNEL_GPU_CUDA_SRC})
    set_source_files_properties(
        src/akari/kernel/integrators/gpu/cuda/integrator.cpp
        src/akari/kernel/scene.cpp
       src/akari/kernel/cuda/launch.cpp
         PROPERTIES LANGUAGE CUDA
    )
else()
    set(AKR_KERNEL_SRC ${AKR_KERNEL_COMMON_SRC} ${AKR_KERNEL_CPU_SRC})
endif()

add_library(akari_core_base SHARED ${AKR_COMMON_SRC}  ${AKR_CORE_SRC} ${AKR_CONFIG_H}  ${AKR_SHADING_SRC})
add_dependencies(akari_core_base akari-configure)
target_link_libraries(akari_core_base PUBLIC  ${AKR_CORE_EXT_LIBS} ${EMBREE_LIBRARY} ${AKR_CUDA_LIBS})
if(AKR_ENABLE_PYTHON)
    target_link_libraries(akari_core_base ${Python_LIBRARIES})
endif()
target_compile_options(akari_core_base PUBLIC ${AKR_CXX_FLAGS})
target_compile_definitions(akari_core_base PUBLIC ${AKR_COMPILE_DEFINITIONS})
target_include_directories(akari_core_base PUBLIC src/)
target_include_directories(akari_core_base PUBLIC external/pybind11/include)
target_include_directories(akari_core_base PUBLIC ${Python_INCLUDE_DIRS} ${EMBREE_INCLUDE_DIRS})
if(AKR_ENABLE_PYTHON)
    target_include_directories(akari_core_base ${Python_INCLUDE_DIRS})
endif()
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL  "3.16")
if(NOT WIN32)
    target_precompile_headers(akari_core_base
        PUBLIC ${PYBIND11_HEADERS}
        <vector>
        <string>
        <unordered_map>
        <memory>
        <thread>
        <atomic>)
endif()

    set_target_properties(akari_core_base PROPERTIES UNITY_BUILD ON)
endif()
set_target_CUDA_props(akari_core_base)


set(AKR_KERNEL_SOA_H ${CMAKE_CURRENT_SOURCE_DIR}/src/akari/kernel/soa.h)
add_soa_header(${AKR_KERNEL_SOA_H} src/akari/kernel/akari.soa)

set(AKR_KERNEL_WORKITEM_SOA_H ${CMAKE_CURRENT_SOURCE_DIR}/src/akari/kernel/integrators/gpu/workitem-soa.h)
add_soa_header(${AKR_KERNEL_WORKITEM_SOA_H} src/akari/kernel/integrators/gpu/workitem.soa)

add_library(akari_core SHARED ${AKR_COMMON_SRC} ${AKR_PYBIND_SRC} ${AKR_KERNEL_SRC}  ${AKR_KERNEL_SOA_H} ${AKR_KERNEL_WORKITEM_SOA_H})
target_link_libraries(akari_core PUBLIC akari_core_base) 
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16")
    set_target_properties(akari_core PROPERTIES UNITY_BUILD ON)
endif()
set_target_CUDA_props(akari_core)



add_executable(akari src/akari/cmd/akari.cpp ${AKR_CONFIG_H})
target_include_directories(akari PUBLIC src/)
target_link_libraries(akari akari_core)
set_target_CUDA_props(akari)



add_executable(akari-import src/akari/cmd/akari-import.cpp ${AKR_CONFIG_H})
target_include_directories(akari-import PUBLIC src/)
target_link_libraries(akari-import akari_core)
set_target_CUDA_props(akari-import)



add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
file(GLOB AKR_TEST src/akari/tests/*.*)
add_executable(akari-test ${AKR_TEST} src/akari/cmd/akari-test.cpp)
target_link_libraries(akari-test gtest akari_core)
add_test(NAME akari COMMAND akari-test)
set_target_CUDA_props(akari-test)

