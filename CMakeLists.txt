cmake_minimum_required(VERSION 3.12)
project(AkariRender)

set(CMAKE_CXX_STANDARD 17)
option(AKARI_BUILD_GUI "Build Gui / Editor" OFF)
option(AKARI_PYTHON_SUPPORT "Enable Python Support" OFF)

include_directories(external/glm)
include_directories(external/misc)
include_directories(external/json)
include_directories(external/cxxopts/include)
include_directories(external/magic_enum/include)

include_directories(include)

set(AKARI_CORE_EXT_LIBS "")
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /MD  /MP /GL /arch:AVX2")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /LTCG:INCREMENTAL")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} /LTCG:INCREMENTAL")
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} /LTCG:INCREMENTAL")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -mavx2")
    set(AKARI_CORE_EXT_LIBS ${AKARI_CORE_EXT_LIBS} stdc++fs)
endif()

if(NOT WIN32)
    set(AKARI_CORE_EXT_LIBS ${AKARI_CORE_EXT_LIBS} pthread dl)
endif()


#
#if(AKARI_BUILD_GUI)
#    include_directories(external/glfw-3.3.2/include)
#    add_subdirectory(external/glfw-3.3.2)
#    set(AKARI_CORE_EXT_LIBS ${AKARI_CORE_EXT_LIBS} glfw)
#endif()
#
include_directories(external/tinyobjloader/)
add_library(tinyobjloader external/tinyobjloader/tiny_obj_loader.cc)
set(AKARI_CORE_EXT_LIBS ${AKARI_CORE_EXT_LIBS} tinyobjloader)



include_directories(external/fmt-6.1.2/include)
add_subdirectory(external/fmt-6.1.2)
set(AKARI_CORE_EXT_LIBS ${AKARI_CORE_EXT_LIBS} fmt)

include_directories(external/asmjit/src)
add_subdirectory(external/asmjit)

if(AKARI_BUILD_GUI)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    include_directories(external/glfw-3.3.2/include)
    add_subdirectory(external/glfw-3.3.2)
    include(cmake/GUI.cmake)
endif()

include(cmake/Embree.cmake)
include(cmake/OIDN.cmake)
#include(cmake/LLVM.cmake)

##set(AKARI_CORE_EXT_LIBS ${AKARI_CORE_EXT_LIBS} embree embree_sse42 embree_avx embree_avx2 lexers math simd tasking sys)
#
file(GLOB AKARI_API_SRC include/akari/core/*.* include/akari/render/*.*)
file(GLOB AKARI_CORE_SRC src/Core/*.* )
file(GLOB AKARI_CORE_RENDER_SRC src/Render/*.*)
add_library(AkariCore SHARED ${AKARI_CORE_SRC})
target_link_libraries(AkariCore  ${AKARI_CORE_EXT_LIBS})
add_library(AkariCoreRender SHARED ${AKARI_CORE_RENDER_SRC})
target_link_libraries(AkariCoreRender AkariCore)
#
#
set(AKARI_ALL_PLUGINS "")
set(PYTHON_INTERPRETER "python3")

function(add_plugin target)
    list(SUBLIST ARGN 0 -1 FILES)
    list(GET FILES 0 MAIN_FILE)
#    list(TRANSFORM FILES PREPEND "${CMAKE_CURRENT_BINARY_DIR}/generated-" OUTPUT_VARIABLE GEN_SRC)
#    list(TRANSFORM GEN_SRC APPEND ".hpp")
    set(GEN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/generated/${target}.hpp")
    add_custom_command(
            OUTPUT ${GEN_SRC}
            COMMAND ${PYTHON_INTERPRETER} ${CMAKE_SOURCE_DIR}/src/tool/reflection_gen.py ${target} ${MAIN_FILE} ${GEN_SRC}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${FILES} ${CMAKE_SOURCE_DIR}/src/tool/reflection_gen.py
    )
    add_library(${target} SHARED ${FILES} ${GEN_SRC})
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${target} PROPERTIES PREFIX "")
    set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/"
#            ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
            RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/")
    target_link_libraries(${target} AkariCoreRender)
endfunction()
function(add_tool target)
    list(SUBLIST ARGN 0 -1 FILES)
    add_executable(${target} ${FILES})
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${target} PROPERTIES PREFIX "")
    set_target_properties(${target} PROPERTIES
            #            ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
            RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/")
    target_link_libraries(${target} AkariCoreRender)
endfunction()
function(add_test target)
    list(SUBLIST ARGN 0 -1 FILES)
    add_executable(${target} ${FILES})
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${target} PROPERTIES PREFIX "")
    set_target_properties(${target} PROPERTIES
            #            ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
            RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/")
    target_link_libraries(${target} AkariCore)
endfunction()
add_subdirectory(src/render)
add_subdirectory(src/tool)

add_executable(akari src/app/akari.cpp)
target_link_libraries(akari AkariCoreRender)

#add_executable(AkariExpr src/app/akari_expr.cpp)
#target_link_libraries(AkariExpr asmjit)
#target_link_libraries(AkariExpr AkariCore)
#target_link_libraries(AkariExpr ComplexTexture)
if(AKARI_BUILD_GUI)
    file(GLOB AkariGuiSrc src/app/ui/*.*)
    add_executable(AkariGui ${AkariGuiSrc})
    target_link_libraries(AkariGui ImGui AkariCore)
endif()

add_executable(test-simd src/test/simd.cpp)
target_link_libraries(test-simd AkariCore)

add_subdirectory(src/test)


#add_executable(test-reflect src/test/test-reflect.cpp src/core/reflect.cpp include/akari/core/reflect.hpp)
#target_link_libraries(test-reflect fmt)