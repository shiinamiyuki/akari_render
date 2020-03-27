if(AKARI_BUILD_GUI)
    include_directories(external/gl3w/include)
    add_library(gl3w external/gl3w/gl3w.c)

    include_directories(external/imgui-docking)
    file(GLOB ImGuiSrc external/imgui-docking/*.cpp ../external/imgui-docking/*.h
            external/imgui-docking/examples/imgui_impl_opengl3.cpp
            external/imgui-docking/examples/imgui_impl_glfw.cpp)
    add_library(ImGui ${ImGuiSrc})
    target_link_libraries(ImGui gl3w glfw)
endif()