// MIT License
//
// Copyright (c) 2019 椎名深雪
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "WindowContext.h"
#include <Akari/Core/Logger.h>
#include <Akari/Render/SceneGraph.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef _WIN32
#include <commdlg.h>
#include <thread>
#include <windows.h>
#else
#endif
namespace Akari::Gui {
    using ModalCloseFunc = std::function<void(void)>;
    using ModalClosure = std::function<void(const ModalCloseFunc &)>;

#ifdef _WIN32
    fs::path GetOpenFilePath() {
        CurrentPathGuard _;
        char filename[MAX_PATH];

        OPENFILENAME ofn;
        ZeroMemory(&filename, sizeof(filename));
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = NULL; // If you have a window to center over, put its HANDLE here
        ofn.lpstrFilter = "Text Files\0*.txt\0Any File\0*.*\0";
        ofn.lpstrFile = filename;
        ofn.nMaxFile = MAX_PATH;
        ofn.lpstrTitle = "Select a File, yo!";
        ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;
        GetOpenFileNameA(&ofn);
        fs::path result(ofn.lpstrFile);
        return result;
    }
#else
    fs::path GetOpenFilePath() { return fs::path(); }
#endif
    class MainWindow : public Window {
        using Window::Window;
        std::shared_ptr<SceneGraph> sceneGraph;
        struct LogWindow : LogHandler {
            std::vector<std::pair<LogLevel, std::string>> logs;
            void Clear() { logs.clear(); }
            void Show() {
                if (ImGui::Begin("Console")) {
                    if (ImGui::Button("clear")) {
                        Clear();
                    }
                    ImGui::Separator();
                    for (auto &log : logs) {
                        ImVec4 color(1, 1, 1, 1);
                        ImGui::TextColored(color, "%s", log.second.c_str());
                    }
                }
            }
            void AddMessage(LogLevel level, const std::string &msg) override { logs.emplace_back(level, msg); }
        };
        std::shared_ptr<LogWindow> logWindow;
        ModalClosure _defaultModalClosure = [](const ModalCloseFunc &) {};
        ModalClosure modalClosure = _defaultModalClosure;
        ModalCloseFunc closeFunc = [=]() { modalClosure = _defaultModalClosure; };
        fs::path currentScenePath = fs::current_path();
        fs::path initialWorkingDir = fs::current_path();

      public:
        MainWindow(GLFWContext &ctx) : Window(ctx) {
            ImGuiIO &io = ImGui::GetIO();
            (void)io;
            logWindow = std::make_shared<LogWindow>();
            GetDefaultLogger()->RegisterHandler(logWindow);
        }
        void DoShowSceneGraph() {
            if (!sceneGraph)
                return;
            ;
        }
        void ShowSceneGraph() {
            if (ImGui::Begin("Scene Graph")) {
                DoShowSceneGraph();
                ImGui::End();
            }
        }
        void ShowEditor() {
            ShowSceneGraph();
            logWindow->Show();
        }
        void ShowMenu() {
            if (ImGui::BeginMainMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem("Open")) {
                        Info("open\n");
                        auto path = GetOpenFilePath();
                        if (!path.empty()) {
                            Info("Open {}\n", path.string());
                            std::thread th([=]() {
                                fs::current_path(fs::absolute(path).parent_path());
                                currentScenePath = fs::absolute(fs::current_path());
                                SerializeContext ctx;
                                std::shared_ptr<SceneGraph> graph;
                                {
                                    Info("Loading {}\n", path.string());
                                    std::ifstream in(path);
                                    std::string str((std::istreambuf_iterator<char>(in)),
                                                    std::istreambuf_iterator<char>());
                                    json data = str.empty() ? json::object() : json::parse(str);
                                    graph = std::make_shared<SceneGraph>(Serialize::fromJson<SceneGraph>(ctx, data));
                                }
                                closeFunc();
                            });
                            th.detach();
                            modalClosure = [=](const ModalCloseFunc &close) {
                                ImGui::BeginPopupModal("Loading");
                                ImGui::Text("Please wait");
                                ImGui::Text("%s", path.string().c_str());
                                ImGui::EndPopup();
                            };
                        }
                    }
                    ImGui::EndMenu();
                }
                ImGui::EndMainMenuBar();
            }
        }
        void Render() {}
        void Show() {
            ImGuiIO &io = ImGui::GetIO();
            bool show_demo_window = true;
            bool show_another_window = false;
            ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

            // Main loop
            while (!glfwWindowShouldClose(window)) {
                // Poll and handle events (inputs, window resize, etc.)
                // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use
                // your inputs.
                // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
                // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
                // Generally you may always pass all inputs to dear imgui, and hide them from your application based on
                // those two flags.
                glfwPollEvents();

                // Start the Dear ImGui frame
                ImGui_ImplOpenGL3_NewFrame();
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse
                // its code to learn more about Dear ImGui!).
                if (show_demo_window)
                    ImGui::ShowDemoWindow(&show_demo_window);

                // 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
                {
                    static float f = 0.0f;
                    static int counter = 0;

                    ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

                    ImGui::Text("This is some useful text."); // Display some text (you can use a format strings too)
                    ImGui::Checkbox("Demo Window", &show_demo_window); // Edit bools storing our window open/close state
                    ImGui::Checkbox("Another Window", &show_another_window);

                    ImGui::SliderFloat("float", &f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
                    ImGui::ColorEdit3("clear color", (float *)&clear_color); // Edit 3 floats representing a color

                    if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when
                                                 // edited/activated)
                        counter++;
                    ImGui::SameLine();
                    ImGui::Text("counter = %d", counter);

                    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                                ImGui::GetIO().Framerate);
                    ImGui::End();
                }

                // 3. Show another simple window.
                if (show_another_window) {
                    ImGui::Begin(
                        "Another Window",
                        &show_another_window); // Pass a pointer to our bool variable (the window will have a closing
                    // button that will clear the bool when clicked)
                    ImGui::Text("Hello from another window!");
                    if (ImGui::Button("Close Me"))
                        show_another_window = false;
                    ImGui::End();
                }
                modalClosure(closeFunc);
                ShowMenu();
                ShowEditor();
                // Rendering
                ImGui::Render();
                int display_w, display_h;
                glfwGetFramebufferSize(window, &display_w, &display_h);
                glViewport(0, 0, display_w, display_h);
                glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
                glClear(GL_COLOR_BUFFER_BIT);
                ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

                // Update and Render additional Platform Windows
                // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to
                // paste this code elsewhere.
                //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
                if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
                    GLFWwindow *backup_current_context = glfwGetCurrentContext();
                    ImGui::UpdatePlatformWindows();
                    ImGui::RenderPlatformWindowsDefault();
                    glfwMakeContextCurrent(backup_current_context);
                }

                glfwSwapBuffers(window);
            }
        }
    };

    std::shared_ptr<Window> CreateAppWindow(GLFWContext &context) { return std::make_shared<MainWindow>(context); }

} // namespace Akari::Gui