set(LLVM_ROOT "" CACHE STRING "Root to LLVM")

if(LLVM_ROOT STREQUAL "")
    message(FATAL_ERROR "LLVM_ROOT must be set")
endif()


set(LLVM_LIBRARIES ${LLVM_ROOT}/bin)
set(LLVM_INCLUDE_DIRS ${LLVM_ROOT}/include)