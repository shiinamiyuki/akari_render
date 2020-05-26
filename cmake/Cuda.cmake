option(AKR_ENABLE_GPU "Enable GPU Rendering" OFF)

if(AKR_ENABLE_GPU)
    find_package(CUDA)
endif()