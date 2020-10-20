#----- Build Shaders ------
file(GLOB AKR_COMMON_SHADER_SRC *.glsl)
add_shader_module(AKR_COMMON_SHADER STATIC SOURCE common.cpp SHADER ${AKR_COMMON_SHADER_SRC})
#--------------------------