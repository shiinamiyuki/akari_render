add_plugin(RTAO  ao.cpp)
add_plugin(Path  pt.cpp)
# add_plugin(GuidedPath  ppg.cpp)

if(AKR_ENABLE_CUDA)
    add_plugin(CUDAPathTracer cuda_pt.cpp)
    copy_to_dist(CUDAPathTracer)
endif()