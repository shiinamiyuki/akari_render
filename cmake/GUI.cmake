if(AKR_BUILD_GUI)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL    OFF CACHE BOOL " " FORCE)
    
    add_subdirectory(external/glfw-3.3.2)    
   
    add_library(gl3w external/gl3w/gl3w.c)
  
    target_include_directories(gl3w PUBLIC external/gl3w/include)
    file(GLOB ImGuiSrc external/imgui-docking/*.cpp ../external/imgui-docking/*.h
            external/imgui-docking/examples/imgui_impl_opengl3.cpp
            external/imgui-docking/examples/imgui_impl_glfw.cpp)
    add_library(ImGui ${ImGuiSrc})
    target_include_directories(ImGui PUBLIC external/glfw-3.3.2/include)
    target_include_directories(ImGui PUBLIC external/imgui-docking)
    target_link_libraries(ImGui gl3w glfw)
endif()