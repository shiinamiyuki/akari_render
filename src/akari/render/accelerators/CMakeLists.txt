add_plugin(SBVH  sbvh.cpp sbvh.h)
add_plugin(Embree  embree.cpp)
target_link_libraries(Embree ${EMBREE_LIBRARY})