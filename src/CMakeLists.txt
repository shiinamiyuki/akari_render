
function(set_output_dir target)
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dist"

        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/dist"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/dist"

        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/dist"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/dist"


        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/dist"
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/dist"

    )
endfunction()

function(file2c)
    set(options "")
    set(oneValueArgs TARGET NAMESPACE)
    set(multiValueArgs FILES )
    cmake_parse_arguments(FILE2C "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )
    add_custom_command(COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tools/file2c.py ${FILE2C_FILES} --ns ${FILE2C_NAMESPACE} --source ${CMAKE_CURRENT_BINARY_DIR}/${FILE2C_TARGET}.cpp --header ${CMAKE_CURRENT_BINARY_DIR}/${FILE2C_TARGET}.h
        DEPENDS ${CMAKE_SOURCE_DIR}/tools/file2c.py ${FILE2C_FILES}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${FILE2C_TARGET}.cpp ${CMAKE_CURRENT_BINARY_DIR}/${FILE2C_TARGET}.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    add_library(${FILE2C_TARGET} STATIC ${CMAKE_CURRENT_BINARY_DIR}/${FILE2C_TARGET}.cpp)
    target_include_directories(${FILE2C_TARGET} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

set(AKR_EXT_LIBS "")
set(AKR_EXT_INCLUDES "")
set(AKR_COMPILE_DEFINITIONS "")

find_package(glm REQUIRED)
find_package(cereal REQUIRED)
find_package(assimp REQUIRED)
# find_package(Eigen3 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
# find_package(IlmBase REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(spdlog REQUIRED)
find_package(stb REQUIRED)

if(AKR_USE_OPENVDB)
    set(Boost_ROOT ${CMAKE_SOURCE_DIR}/.useless/source/boost)
    find_package(Boost REQUIRED)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/.useless/install ${CMAKE_SOURCE_DIR}/.useless/install/lib/cmake/OpenVDB)
    find_package(OpenVDB REQUIRED)
    set(AKR_EXT_LIBS ${AKR_EXT_LIBS} OpenVDB::openvdb )
endif()
set(AKR_EXT_LIBS ${AKR_EXT_LIBS} glm cereal::cereal  embree
    spdlog::spdlog spdlog::spdlog_header_only OpenEXR::IlmImf OpenEXR::IlmImfUtil stb_image stb_image_write)

message(STATUS "python:  " ${Python_EXECUTABLE})

if(AKR_BACKEND_EMBREE)
    find_package(embree 3 REQUIRED)
    set(AKR_COMPILE_DEFINITIONS AKR_BACKEND_EMBREE)
    message("Embree found " ${EMBREE_INCLUDE_DIRS})
    set(AKR_EXT_INCLUDES ${AKR_EXT_INCLUDES} ${EMBREE_INCLUDE_DIRS})
endif()

set(AKR_DEFS "")
if(MSVC)
    set(AKR_CXX_FLAGS /std:c++17 /MP /arch:AVX2 /WX)
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ")
    set(CMAKE_STATIC_LINKER_FLAGS  "${CMAKE_STATIC_LINKER_FLAGS} ")
    set(CMAKE_SHARED_LINKER_FLAGS   "${CMAKE_SHARED_LINKER_FLAGS} ")
else()
    set(CMAKE_CXX_FLAGS -fPIC ${CMAKE_CXX_FLAGS})
    set(AKR_CXX_FLAGS -mavx2)
    set(AKR_EXT_LIBS ${AKR_EXT_LIBS} stdc++fs)
endif()

file(GLOB AKR_LIB_SRC akari/*.*)
set(AKR_GPU_LIBS "")
set(AKR_GPU_SRC "")
if(AKR_BACKEND_CUDA)
    include(${CMAKE_SOURCE_DIR}/cmake/bin2c.cmake)
    function(cuda_pp)
        set(options "")
        set(oneValueArgs FILE)
        set(multiValueArgs INCLUDE_DIRS )
        cmake_parse_arguments(CUDA_PP "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )
        list(TRANSFORM CUDA_PP_INCLUDE_DIRS PREPEND -I)
        add_custom_command(COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} --preprocess ${CUDA_PP_FILE} -o ${CUDA_PP_FILE}.pp ${CUDA_PP_INCLUDE_DIRS}
                       DEPENDS ${CUDA_PP_FILE}
                       OUTPUT ${CUDA_PP_FILE}.pp
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    endfunction()
    file(GLOB AKR_GPU_SRC0 akari/gpu/cuda/*.cpp akari/gpu/*.cpp)
    set(CUDA_COMPILE_DEFNS "-O3;--use_fast_math;--expt-relaxed-constexpr")
    add_ptx_embed_library(optix_kernels akari/gpu/cuda/optix.cu
        INCLUDE_DIRECTORIES ${AKR_OPTIX7_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/.useless/install/include
        COMPILE_DEFINITIONS
        ${CUDA_COMPILE_DEFNS}
        # These options don't seem to do print any warnings?
        -Xptxas --warn-on-double-precision-use
        -Xptxas -Werror)
    add_ptx_embed_library(cuda_volpath_kernel akari/gpu/cuda/volpath.cu
        INCLUDE_DIRECTORIES ${AKR_OPTIX7_PATH}/include ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/.useless/install/include
        COMPILE_DEFINITIONS
        ${CUDA_COMPILE_DEFNS}
        # These options don't seem to do print any warnings?
        -Xptxas --warn-on-double-precision-use
        -Xptxas -Werror)
    file(GLOB AKR_PTX_SRC akari/gpu/cuda/ptx/*)
    add_library(cuda_ptx_libs STATIC ${AKR_PTX_SRC})
    target_link_libraries(cuda_ptx_libs PUBLIC optix_kernels cuda_volpath_kernel)
    set(AKR_GPU_SRC ${AKR_GPU_SRC} ${AKR_GPU_SRC0})
    set(AKR_GPU_LIBS ${AKR_GPU_LIBS} CUDA::nvrtc CUDA::cudart_static CUDA::cuda_driver cuda_ptx_libs)
    set(AKR_DEFS ${AKR_DEFS} AKR_BACKEND_CUDA)
endif()
add_library(nano-akari STATIC ${AKR_LIB_SRC} ${AKR_GPU_SRC})
target_link_libraries(nano-akari PUBLIC ${AKR_EXT_LIBS} ${AKR_GPU_LIBS})
target_include_directories(nano-akari PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${AKR_EXT_INCLUDES})
target_compile_options(nano-akari PUBLIC ${AKR_CXX_FLAGS})
target_compile_definitions(nano-akari PUBLIC ${AKR_DEFS})
if(AKR_BACKEND_CUDA)
    target_include_directories(nano-akari PUBLIC ${AKR_OPTIX7_PATH}/include)
endif()
if(AKR_USE_OPENVDB)
    target_compile_definitions(nano-akari PUBLIC AKR_USE_OPENVDB)
endif()
set_output_dir(nano-akari)
set_target_properties(nano-akari PROPERTIES UNITY_BUILD ON)
add_library(pyakari SHARED pyakari/module.cpp pyakari/module.h)
target_link_libraries(pyakari PUBLIC nano-akari  ${AKR_EXT_LIBS} pybind11::lto pybind11::embed pybind11::module)
target_include_directories(pyakari PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${pybind11_INCLUDE_DIRS} ${AKR_EXT_INCLUDES})
if(MSVC)
    target_compile_options(pyakari PUBLIC /bigobj)
endif()
set_output_dir(pyakari)
if(WIN32)
    set_target_properties(pyakari PROPERTIES SUFFIX ".pyd")
endif()
add_library(akari::core ALIAS nano-akari)
add_library(akari::python ALIAS pyakari)


if(AKR_ENABLE_ENGINE)
    add_subdirectory(akari-engine)
endif()
if(AKR_BUILD_DIFF)
    add_subdirectory(diff-akari)
endif()
add_subdirectory(main)