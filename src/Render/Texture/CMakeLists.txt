add_plugin(RGBTexture RGBTexture.cpp)
add_plugin(ComplexTexture ComplexTexture.cpp)
add_plugin(ImageTexture ImageTexture.cpp)

target_link_libraries(ComplexTexture asmjit)