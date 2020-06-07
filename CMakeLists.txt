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


include(cmake/GUI.cmake)
include(cmake/Embree.cmake)
include(cmake/OIDN.cmake)
include(cmake/Vulkan.cmake)
include(cmake/LLVM.cmake)

##set(AKR_CORE_EXT_LIBS ${AKR_CORE_EXT_LIBS} embree embree_sse42 embree_avx embree_avx2 lexers math simd tasking sys)
#
file(GLOB AKR_API_SRC include/akari/core/*.* include/akari/render/*.*)
file(GLOB AKR_CORE_SRC src/core/*.* )
file(GLOB AKR_CORE_RENDER_SRC src/render/*.*)
file(GLOB AKR_COMPUTE_SRC src/compute/*.* src/compute/*/*.*)
add_library(AkariCore SHARED ${AKR_CORE_SRC})
target_link_libraries(AkariCore  ${AKR_CORE_EXT_LIBS})
target_compile_options(AkariCore PUBLIC ${AKR_CXX_FLAGS})
add_library(AkariCoreRender SHARED ${AKR_CORE_RENDER_SRC})
target_link_libraries(AkariCoreRender AkariCore)
#
#
set(AKR_ALL_PLUGINS "")
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

add_library(AkariCompute SHARED ${AKR_COMPUTE_SRC})
target_link_libraries(AkariCompute AkariCore)
target_include_directories(AkariCompute PUBLIC ${LLVM_INCLUDE_DIR})
target_link_directories(AkariCompute PUBLIC ${LLVM_LIBRARY_DIRS})
target_link_libraries(AkariCompute
        LLVMX86Disassembler
        LLVMX86AsmParser
#        LLVMX86AsmPrinter
        LLVMX86CodeGen

        LLVMSelectionDAG

        LLVMAsmPrinter
        LLVMMCParser
        LLVMX86Info

        LLVMMCJIT
        LLVMInterpreter
        LLVMExecutionEngine

        LLVMCodeGen
        LLVMScalarOpts
        LLVMTransformUtils

        LLVMipo
        LLVMAnalysis
        LLVMTarget
        LLVMMC

        LLVMCore
        LLVMSupport
        )


add_executable(akaric src/app/akari_expr.cpp)
target_link_libraries(akaric AkariCompute)

if(AKR_BUILD_GUI)
    file(GLOB AkariGuiSrc src/app/ui/*.*)
    add_executable(AkariGui ${AkariGuiSrc})
    target_link_libraries(AkariGui ImGui AkariCore)
endif()

add_executable(test-simd src/test/simd.cpp)
target_link_libraries(test-simd AkariCore)

add_subdirectory(src/test)


#add_executable(test-reflect src/test/test-reflect.cpp src/core/reflect.cpp include/akari/core/reflect.hpp)
#target_link_libraries(test-reflect fmt)