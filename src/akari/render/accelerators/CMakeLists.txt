add_plugin(SBVH SOURCE sbvh.cpp sbvh.h)
add_plugin(Embree SOURCE embree.cpp)
target_link_libraries(Embree ${EMBREE_LIBRARY})