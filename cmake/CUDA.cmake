set(AKR_ENABLE_CUDA OFF)
include (CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    find_package (CUDAToolKit REQUIRED)
    if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.17.0")
        set (CMAKE_CUDA_STANDARD 17)
    endif ()

    message (STATUS "Found CUDA: ${CUDAToolkit_BIN_DIR}")
    if (CUDA_VERSION_MAJOR LESS 11)
        message(SEND_ERROR "AkariRender requires CUDA version 11.0 or later. If you have multiple versions installed, please update your PATH.")
    endif ()
    enable_language (CUDA)
    
    # FIXME
    include_directories (${CUDAToolkit_INCLUDE_DIRS})  # for regular c++ compiles

    # http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
    set (AKR_CUDA_DIAG_FLAGS "")
    #set (AKR_CUDA_DIAG_FLAGS "${AKR_CUDA_DIAG_FLAGS} -Xptxas --warn-on-double-precision-use")
    set (AKR_CUDA_DIAG_FLAGS "${AKR_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=partial_override")
    set (AKR_CUDA_DIAG_FLAGS "${AKR_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=virtual_function_decl_hidden")
    set (AKR_CUDA_DIAG_FLAGS "${AKR_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=integer_sign_change")
    set (AKR_CUDA_DIAG_FLAGS "${AKR_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=declared_but_not_referenced")
    # WAR invalid warnings about this with "if constexpr"
    set (AKR_CUDA_DIAG_FLAGS "${AKR_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=implicit_return_from_non_void_function")
    set (AKR_CUDA_DIAG_FLAGS "${AKR_CUDA_DIAG_FLAGS} --expt-relaxed-constexpr")
    set (AKR_CUDA_DIAG_FLAGS "${AKR_CUDA_DIAG_FLAGS} --extended-lambda")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${AKR_CUDA_DIAG_FLAGS}")

    # Willie hears yeh..
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xnvlink -suppress-stack-size-warning")



    # https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
    # Get CUDA compute capability
    set (OUTPUTFILE ${CMAKE_BINARY_DIR}/checkcuda)
    if (MSVC)
            execute_process (COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/checkcuda.cu -ccbin ${CMAKE_CXX_COMPILER} -o ${OUTPUTFILE})
    else  ()
            execute_process (COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/checkcuda.cu -o ${OUTPUTFILE})
    endif ()

    execute_process (COMMAND ${OUTPUTFILE}
                    RESULT_VARIABLE CUDA_RETURN_CODE
                    OUTPUT_VARIABLE AKR_CUDA_ARCH)

    # set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17 -rdc=true --keep")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17 -rdc=true --keep --use_fast_math -maxrregcount 128")
    #if (CMAKE_BUILD_TYPE MATCHES Debug)
    #    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -G -g")
    #else()
    #    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo -maxrregcount 128")
    #endif ()

    if (NOT ${CUDA_RETURN_CODE} EQUAL 0)
        message (SEND_ERROR "Unable to determine GPU's compute capability")
    else ()
        message (STATUS "CUDA Architecture: ${AKR_CUDA_ARCH}")
    endif ()
    set(AKR_COMPILE_DEFINITIONS ${AKR_COMPILE_DEFINITIONS} AKR_ENABLE_GPU AKR_GPU_BACKEND_CUDA)
    set(AKR_ENABLE_CUDA ON)
    macro (set_target_CUDA_props target)
        set_target_properties(${target} 
        PROPERTIES  CUDA_ARCHITECTURES ${AKR_CUDA_ARCH}
                    CUDA_SEPARABLE_COMPILATION  ON )
    endmacro()
    set(AKR_CUDA_LIBS CUDA::cudart CUDA::cuda_driver)
    set(AKR_CXX_FLAGS "") # or nvcc chokes

    if(NOT AKR_OPTIX_PATH)
        message(FATAL_ERROR "AKR_OPTIX_PATH must be set")
    endif()
    include_directories(${AKR_OPTIX_PATH}/include)

    macro (cuda_compile_and_embed output_var cuda_file lib_name)
        add_library ("${lib_name}" OBJECT "${cuda_file}")
        set_property (TARGET "${lib_name}" PROPERTY CUDA_PTX_COMPILATION ON)
        target_compile_options ("${lib_name}"
        PRIVATE
            # disable "extern declaration... is treated as a static definition" warning
            -Xcudafe=--display_error_number -Xcudafe=--diag_suppress=3089
            )
        target_compile_definitions("${lib_name}" PRIVATE ${AKR_COMPILE_DEFINITIONS})
        target_include_directories ("${lib_name}" PRIVATE src ${CMAKE_BINARY_DIR})
        # target_include_directories ("${lib_name}" SYSTEM PRIVATE ${NANOVDB_INCLUDE})
        # add_dependencies ("${lib_name}" pbrt_soa_generated)
        set (c_var_name ${output_var})
        set (embedded_file ${cuda_file}.ptx_embedded.c)
        add_custom_command (
        OUTPUT "${embedded_file}"
        COMMAND ${CMAKE_COMMAND}
            "-DBIN_TO_C_COMMAND=${BIN2C}"
            "-DOBJECTS=$<TARGET_OBJECTS:${lib_name}>"
            "-DVAR_NAME=${c_var_name}"
            "-DOUTPUT=${embedded_file}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/bin2c_wrapper.cmake
        VERBATIM
        DEPENDS "${lib_name}"
        COMMENT "compiling (and embedding ptx from) ${cuda_file}"
        )
        set (${output_var} ${embedded_file})
    endmacro ()
else()
    macro (set_target_CUDA_props target)
        set_target_properties(${target} PROPERTIES  CUDA_ARCHITECTURES OFF)
    endmacro()
endif()