
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

add_library(nano-akari STATIC ${AKR_LIB_SRC})
target_link_libraries(nano-akari PUBLIC ${AKR_EXT_LIBS} )
target_include_directories(nano-akari PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${AKR_EXT_INCLUDES})
target_compile_options(nano-akari PUBLIC ${AKR_CXX_FLAGS})
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
add_subdirectory(main)