add_plugin(BVHAccelerator bvh-backend.cpp)

if(AKR_USE_EMBREE)
    add_plugin(EmbreeAccelerator embree-backend.cpp)
    target_include_directories(EmbreeAccelerator PRIVATE ${EMBREE_INCLUDE_DIRS})
    target_link_libraries(EmbreeAccelerator  ${EMBREE_LIBRARY})
endif()