# set(CMAKE_CXX_STANDARD 23)
# set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "")
# set(EMBREE_GEOMETRY_QUAD OFF CACHE BOOL "")
# set(EMBREE_GEOMETRY_SUBDIVISION OFF CACHE BOOL "")
# set(EMBREE_TUTORIALS OFF CACHE BOOL "")
# set(EMBREE_STATIC_LIB ON CACHE BOOL "")
# add_subdirectory(embree)
# # if compiler is clang-cl, disable unsafe-buffer-usage
# if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
#     target_compile_options(embree PRIVATE "-Wunsafe-buffer-usage")
# endif()
# set_target_properties(embree PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

