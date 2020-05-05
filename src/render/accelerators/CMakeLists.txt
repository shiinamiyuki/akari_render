add_plugin(BVHAccelerator bvh-backend.cpp)

if(AKARI_USE_EMBREE)
    add_plugin(EmbreeAccelerator embree-backend.cpp)
    target_link_libraries(EmbreeAccelerator  ${EMBREE_LIBRARY})
endif()