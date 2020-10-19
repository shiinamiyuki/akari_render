cmake_minimum_required(VERSION 3.12)
project(AkariRender)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
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

set(CMAKE_SHARED_LIBRARY_PREFIX_CXX "")
set(CMAKE_STATIC_LIBRARY_PREFIX_CXX "")
set(AKR_CORE_EXT_LIBS "")
set(AKR_COMPILE_DEFINITIONS AKR_ENABLE_CPU)
if(MSVC)
    set(AKR_CXX_FLAGS /std:c++17 /MP  /GL /arch:AVX2 /WX)
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ")
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} ")
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

macro(add_build_util target )
    add_executable(${target} ${ARGN})
    set(${target}_SOURCE ${ARGN})
endmacro()
# ---------------- Begin Build Utils --------------------
# add_build_util(akari-configure src/akari/tools/configure.cpp)
# target_link_libraries(akari-configure fmt)
# set_target_properties(akari-configure 
#         PROPERTIES  CUDA_ARCHITECTURES OFF)

# add_build_util(akari-soac src/akari/tools/soac.cpp)
# target_link_libraries(akari-soac fmt)
# set_target_properties(akari-soac
#         PROPERTIES  CUDA_ARCHITECTURES OFF)
# macro(add_soa_header target source)
#     add_custom_command(OUTPUT ${target}
#         COMMAND akari-soac ${source} ${target}
#         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#         DEPENDS akari-soac ${source} ${CMAKE_SOURCE_DIR}/src/akari/common/soa.h)
# endmacro()

# ---------------- End Build Utils --------------------


include(cmake/Embree.cmake)


# ---------------- Akari Unified Shading Language Compiler
file(GLOB ASL_COMPILER_SRC src/akaric/*.*)
add_build_util(akaric ${ASL_COMPILER_SRC})
target_include_directories(akaric PUBLIC src/)
target_link_libraries(akaric fmt)



function(add_shader_module target)
    list(SUBLIST ARGN 0 -1 sources)
  
    list(APPEND TARGET_SOURCE "")
    foreach(src ${sources} )
    # message(STATUS "src= " ${src})
        add_custom_command(OUTPUT ${src}.h
            COMMAND akaric ${src} -b cpp -o ${src}.h
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${akaric_SOURCE} ${sources})
        list(APPEND TARGET_SOURCE ${src}.h)
    endforeach()
    set_source_files_properties(${TARGET_SOURCE} PROPERTIES LANGUAGE CXX)
    add_library(${target} STATIC ${TARGET_SOURCE})
    # message(STATUS "target= " ${TARGET_SOURCE})
endfunction()


file(GLOB PYBIND11_HEADERS external/pybind11/include/pybind11.h external/pybind11/include/stl.h)
file(GLOB AKR_CORE_SRC src/akari/core/*.*)
file(GLOB AKR_PYBIND_SRC src/akari/core/nodes/*.*)


add_library(libakari_core SHARED ${AKR_COMMON_SRC}  ${AKR_CORE_SRC})
target_link_libraries(libakari_core PUBLIC  ${AKR_CORE_EXT_LIBS} ${EMBREE_LIBRARY} ${AKR_CUDA_LIBS})
if(AKR_ENABLE_PYTHON)
    target_link_libraries(libakari_core ${Python_LIBRARIES})
endif()
target_compile_options(libakari_core PUBLIC ${AKR_CXX_FLAGS})
target_compile_definitions(libakari_core PUBLIC ${AKR_COMPILE_DEFINITIONS})
target_include_directories(libakari_core PUBLIC src/)
target_include_directories(libakari_core PUBLIC external/pybind11/include)
target_include_directories(libakari_core PUBLIC ${Python_INCLUDE_DIRS} ${EMBREE_INCLUDE_DIRS})
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



#----- Build Shaders ------
file(GLOB AKR_COMMON_SHADER_SRC src/akari/render/*.glsl)
add_shader_module(AKR_COMMON_SHADER  ${AKR_COMMON_SHADER_SRC})
#--------------------------

file(GLOB AKR_RENDER_SRC src/akari/render/*.* src/akari/render/kernel/akari_kernel_ispc.h)
add_library(libakari_render SHARED ${AKR_RENDER_SRC})
add_dependencies(libakari_render AKR_COMMON_SHADER)
target_include_directories(libakari_render PUBLIC src/)
target_link_libraries(libakari_render PUBLIC libakari_core)

function(add_plugin target)
    list(SUBLIST ARGN 0 -1 FILES)
    add_library(${target} SHARED ${FILES})
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${target} PROPERTIES PREFIX "")
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
        ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
        RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/plugins/")
    target_link_libraries(${target} libakari_core)
endfunction()
add_subdirectory(src/akari/render)

# set(AKR_KERNEL_SOA_H ${CMAKE_CURRENT_SOURCE_DIR}/src/akari/kernel/soa.h)
# add_soa_header(${AKR_KERNEL_SOA_H} src/akari/kernel/akari.soa)

# set(AKR_KERNEL_WORKITEM_SOA_H ${CMAKE_CURRENT_SOURCE_DIR}/src/akari/kernel/integrators/gpu/workitem-soa.h)
# add_soa_header(${AKR_KERNEL_WORKITEM_SOA_H} src/akari/kernel/integrators/gpu/workitem.soa)

# add_library(libakari_core SHARED ${AKR_COMMON_SRC} ${AKR_PYBIND_SRC} ${AKR_KERNEL_SRC}  ${AKR_KERNEL_SOA_H} ${AKR_KERNEL_WORKITEM_SOA_H})
# target_link_libraries(libakari_core PUBLIC akari_core_base) 
# if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.16")
#     set_target_properties(libakari_core PROPERTIES UNITY_BUILD ON)
# endif()



add_executable(akari src/akari/cmd/akari.cpp )
target_include_directories(akari PUBLIC src/)
target_link_libraries(akari libakari_render)




add_executable(akari-import src/akari/cmd/akari-import.cpp )
target_include_directories(akari-import PUBLIC src/)
target_link_libraries(akari-import libakari_core)


set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(external/googletest EXCLUDE_FROM_ALL)
file(GLOB AKR_TEST src/akari/tests/*.*)
add_executable(akari-test ${AKR_TEST} src/akari/cmd/akari-test.cpp)
target_link_libraries(akari-test gtest libakari_core)
add_test(NAME akari COMMAND akari-test)


