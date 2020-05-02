add_plugin(BVHAccelerator bvh-_backend.cpp)

if(AKARI_USE_EMBREE)
    add_plugin(EmbreeAccelerator embree-_backend.cpp)
    target_link_libraries(EmbreeAccelerator  ${EMBREE_LIBRARY})
endif()