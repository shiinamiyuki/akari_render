file(GLOB SRC *.cpp)
add_library(imgui ${SRC})
target_include_directories(imgui PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(imgui PUBLIC  Vulkan::Vulkan glfw)