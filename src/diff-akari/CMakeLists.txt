# find_package(Enoki REQUIRED)
# find_package(Torch REQUIRED)
# add_executable(test-enoki-nn test-enoki-nn.cpp)
# set_output_dir(test-enoki-nn)
# target_link_libraries(test-enoki-nn PRIVATE ${TORCH_LIBRARIES})
# target_compile_options(test-enoki-nn PRIVATE ${TORCH_CXX_FLAGS})

# if (MSVC)
#     file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#     add_custom_command(TARGET test-enoki-nn
#                     POST_BUILD
#                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                     ${TORCH_DLLS}
#                     $<TARGET_FILE_DIR:test-enoki-nn)
# endif (MSVC)