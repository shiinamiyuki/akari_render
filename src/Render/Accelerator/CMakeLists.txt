add_plugin(BVHAccelerator BVH-Backend.cpp)

if(AKARI_USE_EMBREE)
    add_plugin(EmbreeAccelerator Embree-Backend.cpp)
    target_link_libraries(EmbreeAccelerator  ${EMBREE_LIBRARY})
endif()