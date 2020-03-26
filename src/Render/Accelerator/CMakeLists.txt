add_plugin(BVHAccelerator BVH-Backend.cpp)
add_plugin(EmbreeAccelerator Embree-Backend.cpp)

target_link_libraries(EmbreeAccelerator embree_avx2 lexers math simd tasking sys)