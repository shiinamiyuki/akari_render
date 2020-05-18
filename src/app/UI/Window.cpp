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
#include "window_context.h"

#include <map>
#include "editor_functions.hpp"
#include <akari/core/logger.h>
#include <akari/render/scene_graph.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef _WIN32
#include <commdlg.h>
#include <thread>
#include <windows.h>
#else
#endif
namespace akari::Gui {
    using ModalCloseFunc = std::function<void(void)>;
    using ModalClosure = std::function<void(const ModalCloseFunc &)>;
    struct EditorState {};
    struct ImGuiIdGuard {
        template <typename T> explicit ImGuiIdGuard(T p) { ImGui::PushID(p); }
        ~ImGuiIdGuard() { ImGui::PopID(); }
    };
    namespace detail {
        inline bool EditItem(EditorState &state, const char *label, const Any &ref);
        template <typename T> bool Edit(EditorState &, const char *label, T &value) {
            ImGuiIdGuard _(&value);
            return Gui::Edit(label, value);
        }
        inline bool display_props(EditorState &state,const char * label, const std::shared_ptr<Component>& comp){
             bool ret = false;
             if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
                auto props = Type::get_by_typeid(*comp).get_properties();
                for (auto &prop : props) {
                    ret = ret | EditItem(state, prop.name(), prop.get(make_any_ref(*comp)));
                }
                ImGui::TreePop();
            }
            return ret;
        }
        template<typename T>
        bool choose_implementation(const char * label, std::shared_ptr<T> &value){
            Type type(type_of<T>());
            ImGuiIdGuard _(label);
            std::map<std::string_view, Type> derives;
            type.foreach_derived([&](const Type & derived){
                // derives[derived.pretty_name()] = derived;
                derives.emplace(derived.pretty_name(), derived);
            });
            std::string_view current_select = "None";
            bool changed = false;
            if(value){
                current_select = Type(type_of(*value)).pretty_name();
            }
            if (ImGui::BeginCombo(label, current_select.data())) 
            {
                // for (int n = 0; n < IM_ARRAYSIZE(items); n++)
                // {
                //     bool is_selected = (item_current == items[n]);
                //     if (ImGui::Selectable(items[n], is_selected))
                //         item_current = items[n];
                //     if (is_selected)
                //         ImGui::SetItemDefaultFocus(); 
                // }
                for(auto & item : derives){
                    bool is_selected;
                    if(value == nullptr){ 
                        is_selected = false;
                    }else{
                        is_selected = item.first == current_select;
                    }
                    if (ImGui::Selectable(item.first.data(), is_selected) && current_select != item.first){
                        current_select = item.first;
                        changed = true;
                        
                    }
                }
                ImGui::EndCombo();
            }
            if(changed){
                if(current_select != "None")
                    value = derives.at(current_select).create_shared().shared_cast<T>();
                else{
                    value = nullptr;
                }
            }
            return true;

        }
        template <> inline bool Edit(EditorState &state, const char *label, std::shared_ptr<Material> &value) {
            bool ret = false;
            ImGuiIdGuard _(&value);
            ret = ret | choose_implementation("material", value);
            ret = ret |display_props(state, label, value);
            return ret;
        }
        template <> inline bool Edit(EditorState &state, const char *label, std::shared_ptr<Texture> &value) {
            bool ret = false;
            ImGuiIdGuard _(&value);
            ret = ret | choose_implementation("texture", value);
            ret = ret | display_props(state, label, value);
            return ret;
        }
        template <> inline bool Edit(EditorState &state, const char *label, std::shared_ptr<Mesh> &value) {
            bool ret = false;
            ImGuiIdGuard _(value.get());
            if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
                if (value) {
                    auto &materials = value->GetMaterials();
                    for (auto &material : materials) {
                        std::string _id =
                            fmt::format("{}##{}", material.name.c_str(), (size_t)(const void *)(&material));
                        if (ImGui::TreeNode(_id.c_str())) {
                            ret = ret | Edit(state, "material", material.material);
                            if (ImGui::TreeNodeEx("emission", ImGuiTreeNodeFlags_DefaultOpen)) {
                                ret = ret | Edit(state, "color", material.emission.color);
                                ret = ret | Edit(state, "strength", material.emission.strength);
                                ImGui::TreePop();
                            }
                            ImGui::TreePop();
                        }
                    }
                }
                ImGui::TreePop();
            }
            return ret;
        }

        template <> inline bool Edit(EditorState &state, const char *label, MeshWrapper &value) {
            bool ret = false;
            ImGuiIdGuard _(&value);
            if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("File: %s", value.file.string().c_str());
                ret = ret | Edit(state, "transform", value.transform);
                ret = ret | Edit(state, "mesh", value.mesh);
                ImGui::TreePop();
            }
            return ret;
        }

        template <typename T>
        std::pair<bool, bool> _EditItemV(EditorState &state, const char *label, const Any &reference) {
            if (reference.is_of<T>()) {
                return {true, Edit(state, label, reference.as<T>())};
            }
            return {false, false};
        }

        inline std::pair<bool, bool> EditItemV(EditorState &state, const char *label, const Any &reference) {
            return {false, false};
        }

        template <typename T, typename... Args>
        inline std::pair<bool, bool> EditItemV(EditorState &state, const char *label, const Any &reference) {
            auto [taken, modified] = _EditItemV<T>(state, label, reference);
            if (!taken) {
                if constexpr (sizeof...(Args) > 0) {
                    return EditItemV<Args...>(state, label, reference);
                } else {
                    return {false, false};
                }
            }
            return {taken, modified};
        }

        inline bool EditItem(EditorState &state, const char *label, const Any &ref) {
            return EditItemV<int, float, ivec2, ivec3, vec2, vec3, Spectrum, Angle<float>, Angle<vec3>, TransformManipulator,
                             std::shared_ptr<Texture>, std::shared_ptr<Material>, MeshWrapper>(state, label, ref)
                .second;
        }
    }; // namespace detail

#ifdef _WIN32
    fs::path GetOpenFilePath() {
        CurrentPathGuard _;
        char filename[MAX_PATH];

        OPENFILENAME ofn;
        ZeroMemory(&filename, sizeof(filename));
        ZeroMemory(&ofn, sizeof(ofn));
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = NULL; // If you have a window to center over, put its HANDLE here
        ofn.lpstrFilter = "Text Files\0*.txt\0JSON File\0*.json\0Any File\0*.*\0";
        ofn.lpstrFile = filename;
        ofn.nMaxFile = MAX_PATH;
        ofn.lpstrTitle = "Select a File";
        ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;
        GetOpenFileNameA(&ofn);
        fs::path result(ofn.lpstrFile);
        return result;
    }
#else
    fs::path GetOpenFilePath() { return fs::path(); }
#endif
    class MainWindow : public Window {
        struct WindowFlags {
            bool showStyleEditor = false;
        };
        WindowFlags flags;
        EditorState editorState;
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
                        ImVec4 color = ImGui::GetStyleColorVec4(ImGuiCol_Text);
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
        // std::weak_ptr<Component> selectComponent;
        Any selectedItem;

      public:
        explicit MainWindow(GLFWContext &ctx) : Window(ctx) {
            ImGuiIO &io = ImGui::GetIO();
            (void)io;
            logWindow = std::make_shared<LogWindow>();
            GetDefaultLogger()->RegisterHandler(logWindow);
        }
        void DoShowSceneGraph() {
            if (!sceneGraph)
                return;
            auto &scene = sceneGraph->scene;
            if (ImGui::TreeNode("Meshes")) {
                for (auto &mesh : scene.meshes) {
                    auto name = fs::relative(mesh.file, currentScenePath);
                    static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow |
                                                           ImGuiTreeNodeFlags_OpenOnDoubleClick |
                                                           ImGuiTreeNodeFlags_SpanAvailWidth;
                    if (ImGui::TreeNodeEx(mesh.mesh.get(), base_flags, "%s", name.string().c_str())) {
                        if (ImGui::IsItemClicked()) {
                            debug("selected\n");
                            selectedItem = make_any_ref(mesh);
                        }
                        //  auto props = mesh.mesh->GetProperties();
                        ImGui::TreePop();
                    }
                }
                ImGui::TreePop();
            }
        }
        void ShowSceneGraph() {
            if (ImGui::Begin("Scene Graph")) {
                DoShowSceneGraph();
                ImGui::End();
            }
        }
        template <typename F> void DockSpace(F &&f) {
            ImGuiViewport *viewport = ImGui::GetMainViewport();
            ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
            ImGui::SetNextWindowPos(viewport->GetWorkPos());
            ImGui::SetNextWindowSize(viewport->GetWorkSize());
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                            ImGuiWindowFlags_NoMove;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
            ImGui::Begin("DockSpace", nullptr, window_flags);
            ImGui::PopStyleVar();

            ImGui::PopStyleVar(2);
            ImGuiID dockspace_id = ImGui::GetID("MainDockSpace");
            static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
            f();
            ImGui::End();
        }
        bool EditItem(const char *label, Any &ref) { return detail::EditItem(editorState, label, ref); }
        void ShowInspector() {
            if (ImGui::Begin("Inspector")) {
                if (selectedItem.has_value()) {
                    EditItem("Selected", selectedItem);
                }
                ImGui::End();
            }
        }
        void ShowStyleEditor() {
            if (flags.showStyleEditor && ImGui::Begin("Style Editor", &flags.showStyleEditor)) {
                ImGui::ShowStyleEditor();
                ImGui::End();
            }
        }
        void ShowEditor() {
            ImGuiIO &io = ImGui::GetIO();
            (void)io;
            DockSpace([=]() {
                ShowMenu();
                ShowSceneGraph();
                ShowInspector();
                ShowStyleEditor();
                logWindow->Show();
            });
        }

        void ShowMenu() {
            if (ImGui::BeginMenuBar()) {
                if (ImGui::BeginMenu("File")) {
                    if (ImGui::MenuItem("Open")) {
                        info("open\n");
                        auto path = GetOpenFilePath();
                        if (!path.empty()) {
                            info("Open {}\n", path.string());
                            auto _tmp = sceneGraph;
                            std::thread th([=]() {
                                fs::current_path(fs::absolute(path).parent_path());
                                currentScenePath = fs::absolute(fs::current_path());
                                auto npath = path.filename();
                                try {
                                    info("Loading {}\n", path.string());
                                    std::ifstream in(npath);
                                    std::string str((std::istreambuf_iterator<char>(in)),
                                                    std::istreambuf_iterator<char>());
                                    json data = str.empty() ? json::object() : json::parse(str);
                                    sceneGraph =
                                        std::make_shared<SceneGraph>(serialize::load_from_json<SceneGraph>(data));
                                } catch (std::exception &e) {
                                    error("Exception while loading: {}\n", e.what());
                                    sceneGraph = _tmp;
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
                if (ImGui::BeginMenu("Preferences")) {
                    ImGui::MenuItem("Style", nullptr, &flags.showStyleEditor);
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }
        }
        void Render() {}
        void Show() {
            ImGuiIO &io = ImGui::GetIO();
            bool show_demo_window = true;
            bool show_another_window = false;
            (void)show_another_window;
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

                if (show_demo_window)
                    ImGui::ShowDemoWindow(&show_demo_window);

                modalClosure(closeFunc);

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

} // namespace akari::Gui