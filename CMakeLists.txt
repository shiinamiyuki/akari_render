cmake_minimum_required(VERSION 3.13)
project(AkariRender)

set(CMAKE_CXX_STANDARD 17)
option(AKARI_BUILD_GUI "Build Gui / Editor" OFF)

#
#
include_directories(external/glm)
include_directories(external/misc)
include_directories(external/miyuki.serialize/include)
include_directories(external/cxxopts/include)

include_directories(include)

set(AKARI_CORE_EXT_LIBS "")
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /MD  /MP /GL ")
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /LTCG:INCREMENTAL")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} /LTCG:INCREMENTAL")
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} /LTCG:INCREMENTAL")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
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
##set(AKARI_CORE_EXT_LIBS ${AKARI_CORE_EXT_LIBS} embree embree_sse42 embree_avx embree_avx2 lexers math simd tasking sys)
#
file(GLOB AKARI_API_SRC include/Akari/Core/*.* include/Akari/Render/*.*)
file(GLOB AKARI_CORE_SRC src/Core/*.* src/Render/*.*)
add_library(AkariCore SHARED ${AKARI_CORE_SRC})
target_link_libraries(AkariCore  ${AKARI_CORE_EXT_LIBS})
#
#
set(AKARI_ALL_PLUGINS "")


function(add_plugin target)
    list(SUBLIST ARGN 0 -1 FILES)
    add_library(${target} SHARED ${FILES})
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${target} PROPERTIES PREFIX "")
    set_target_properties(${target} PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/"
#            ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
            RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/")
    target_link_libraries(${target} AkariCore)
endfunction()
function(add_tool target)
    list(SUBLIST ARGN 0 -1 FILES)
    add_executable(${target} ${FILES})
    set_property(TARGET ${target} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${target} PROPERTIES PREFIX "")
    set_target_properties(${target} PROPERTIES
            #            ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/plugins/"
            RUNTIME_OUTPUT_DIRECTORY   "${CMAKE_BINARY_DIR}/")
    target_link_libraries(${target} AkariCore)
endfunction()

add_subdirectory(src/Render)
add_subdirectory(src/Tool)

add_executable(Akari src/App/Akari.cpp)
target_link_libraries(Akari AkariCore)

if(AKARI_BUILD_GUI)
    file(GLOB AkariGuiSrc src/App/UI/*.*)
    add_executable(AkariGui ${AkariGuiSrc})
    target_link_libraries(AkariGui ImGui AkariCore)
endif()


