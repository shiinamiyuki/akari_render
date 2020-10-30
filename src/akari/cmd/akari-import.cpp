// MIT License
//
// Copyright (c) 2020 椎名深雪
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>
#include <regex>
#include <cxxopts.hpp>
#include <akari/core/logger.h>
#include <akari/core/mesh.h>
#include <akari/core/misc.h>
using namespace akari;
std::shared_ptr<Mesh> load_wavefront_obj(const fs::path &path, std::string &generated) {
    info("loading {}", fs::absolute(path).string());
    std::shared_ptr<Mesh> mesh;
    fs::path parent_path = fs::absolute(path).parent_path();
    fs::path file = path.filename();
    CurrentPathGuard _;
    if (!parent_path.empty())
        fs::current_path(parent_path);

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> obj_materials;

    std::string err;

    std::string _file = file.string();
    std::ostringstream os;
    bool ret = misc::LoadObj(&attrib, &shapes, &obj_materials, &err, _file.c_str());
    if (!ret) {
        error("error: {}\n", err);
        return nullptr;
    }

    astd::pmr::vector<float> vertices, normals, texcoords;
    astd::pmr::vector<int> indices;
    astd::pmr::vector<int> material_indices;
    vertices.resize(attrib.vertices.size());
    std::memcpy(&vertices[0], &attrib.vertices[0], sizeof(float) * vertices.size());

    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {

            // auto mat = materials[shapes[s].mesh.material_ids[f]].name;
            int fv = shapes[s].mesh.num_face_vertices[f];
            vec3 triangle[3];
            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                indices.push_back(idx.vertex_index);
                triangle[v] = vec3(load<vec3>(&vertices[3 * idx.vertex_index]));
            }
            material_indices.emplace_back(shapes[s].mesh.material_ids[f]);
            vec3 ng = normalize(cross(triangle[1] - triangle[0], triangle[2] - triangle[0]));
            for (int v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                if (idx.normal_index < 0) {
                    normals.emplace_back(ng[0]);
                    normals.emplace_back(ng[1]);
                    normals.emplace_back(ng[2]);
                } else {
                    normals.emplace_back(attrib.normals[3 * idx.normal_index + 0]);
                    normals.emplace_back(attrib.normals[3 * idx.normal_index + 1]);
                    normals.emplace_back(attrib.normals[3 * idx.normal_index + 2]);
                }
                if (idx.texcoord_index < 0) {
                    texcoords.emplace_back(v > 0);
                    texcoords.emplace_back(v % 2 == 0);
                } else {
                    texcoords.emplace_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
                    texcoords.emplace_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
                }
            }
            index_offset += fv;
        }
    }
    auto normalize_name = [](const std::string &src) {
        auto out = std::regex_replace(src, std::regex("-|\\."), "_");
        return out;
    };
    for (auto &obj_mat : obj_materials) {
        auto normalized = normalize_name(obj_mat.name);
        auto kd = load<vec3>(obj_mat.diffuse);
        auto ks = load<vec3>(obj_mat.specular);
        auto ke = load<vec3>(obj_mat.emission);
        if (hmax(ke) > 0.001) {
            os << "// OBJ Material: " << obj_mat.name << "\n";
            os << "export " << normalized << " = EmissiveMaterial {\n";
            os << "  color : [" << ke[0] << "," << ke[1] << "," << ke[2] << "]\n}\n";
            os << "// ======================================== \n\n";
            continue;
        }
        auto roughness = std::sqrt(2.0f / (obj_mat.shininess + 2.0f));
        float frac = hmax(ks) / (hmax(kd) + hmax(ks));
        if (!std::isnormal(frac)) {
            frac = 0;
        }
        fs::path diffuse_tex = obj_mat.diffuse_texname, specular_tex = obj_mat.specular_texname;

        if (!diffuse_tex.empty() || !specular_tex.empty()) {
            frac = 0.5f;
            os << "// OBJ Material: " << obj_mat.name << " Diffuse part\n";
            os << "let " << normalized << "_diffuse"
               << " = DiffuseMaterial {\n";
            if (diffuse_tex.empty()) {
                os << "  color : [" << obj_mat.diffuse[0] << "," << obj_mat.diffuse[1] << "," << obj_mat.diffuse[2]
                   << "]\n";
            } else {
                os << "  color : \"" << diffuse_tex.generic_string() << "\"\n";
            }
            os << "}\n";
            os << "// OBJ Material: " << obj_mat.name << " Diffuse part\n";
            os << "let " << normalized << "_glossy"
               << " = GlossyMaterial {\n";
            if (specular_tex.empty()) {
                os << "  color : [" << obj_mat.specular[0] << "," << obj_mat.specular[1] << "," << obj_mat.specular[2]
                   << "],\n";
            } else {
                os << "  color : \"" << specular_tex.generic_string() << "\"\n";
            }
            os << "  roughness: " << roughness << "\n}\n";
            os << "export " << normalized << " = MixMaterial {\n";
            os << "  fraction: " << frac << ",\n";
            os << "  first: $" << normalized << "_diffuse"
               << ",\n";
            os << "  second: $" << normalized << "_glossy\n"
               << "}\n";
            os << "// ======================================== \n\n";
        } else if (hmax(ks) < 0.0001f) {
            os << "// OBJ Material: " << obj_mat.name << "\n";
            os << "export " << normalized << " = DiffuseMaterial {\n";
            os << "  color : [" << obj_mat.diffuse[0] << "," << obj_mat.diffuse[1] << "," << obj_mat.diffuse[2]
               << "]\n}\n";
            os << "// ======================================== \n\n";
        } else if (hmax(kd) < 0.0001f) {
            os << "// OBJ Material: " << obj_mat.name << "\n";
            os << "export " << normalized << " = GlossyMaterial {\n";
            os << "  color : [" << obj_mat.specular[0] << "," << obj_mat.specular[1] << "," << obj_mat.specular[2]
               << "],\n  roughness: " << roughness << "\n}\n";
            os << "// ======================================== \n\n";
        } else {
            os << "// OBJ Material: " << obj_mat.name << " Diffuse part\n";
            os << "let " << normalized << "_diffuse"
               << " = DiffuseMaterial {\n";
            os << "  color : [" << obj_mat.diffuse[0] << "," << obj_mat.diffuse[1] << "," << obj_mat.diffuse[2]
               << "]\n}\n";
            os << "// OBJ Material: " << obj_mat.name << " Diffuse part\n";
            os << "let " << normalized << "_glossy"
               << " = GlossyMaterial {\n";
            os << "  color : [" << obj_mat.specular[0] << "," << obj_mat.specular[1] << "," << obj_mat.specular[2]
               << "],\n  roughness: " << roughness << "\n}\n";
            os << "export " << normalized << " = MixMaterial {\n";
            os << "  fraction: " << frac << ",\n";
            os << "  first: $" << normalized << "_diffuse"
               << ",\n";
            os << "  second: $" << normalized << "_glossy\n"
               << "}\n";
            os << "// ======================================== \n\n";
        }
    }
    info("loaded {} triangles, {} vertices", indices.size() / 3, vertices.size() / 3);
    os << "export mesh = AkariMesh {\n  path: \"" << path.filename().concat(".mesh").generic_string() << "\",\n";
    os << "  materials: [\n";
    int idx = 0;
    for (auto &obj_mat : obj_materials) {
        os << "    // material index " << idx << "\n";
        os << "    $" << normalize_name(obj_mat.name) << ",\n";
        idx++;
    }
    os << "  ]\n}\n";
    generated = os.str();
    mesh = std::make_shared<Mesh>(Allocator<>());
    mesh->indices = indices;
    mesh->normals = normals;
    mesh->vertices = vertices;
    mesh->texcoords = texcoords;
    mesh->material_indices = material_indices;
    return mesh;
}

int main(int argc, const char **argv) {
    try {
        cxxopts::Options options("akari-import", " - Import OBJ meshes to akari scene description file");
        options.positional_help("input output").show_positional_help();
        {
            auto opt = options.allow_unrecognised_options().add_options();
            opt("i,input", "Input Scene Description File", cxxopts::value<std::string>());
            opt("o,output", "Input Scene Description File", cxxopts::value<std::string>());
            opt("help", "Show this help");
        }
        options.parse_positional({"input", "output"});
        auto result = options.parse(argc, argv);
        if (!result.count("input")) {
            fatal("Input file must be provided\n");
            std::cout << options.help() << std::endl;
            exit(0);
        }
        if (!result.count("output")) {
            fatal("Output file must be provided\n");
            std::cout << options.help() << std::endl;
            exit(0);
        }
        if (result.arguments().empty() || result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        auto inputFilename = result["input"].as<std::string>();
        auto outputFilename = result["output"].as<std::string>();
        auto generated = std::string();
        auto mesh = load_wavefront_obj(fs::path(inputFilename), generated);
        auto res = std::make_shared<BinaryGeometry>(mesh);
        auto mesh_file = fs::path(inputFilename).filename().concat(".mesh");
        auto out_dir = fs::absolute(outputFilename).parent_path();
        res->save(fs::path(out_dir / mesh_file));
        std::ofstream os(outputFilename);
        os << generated;
    } catch (std::exception &e) {
        fatal("Exception: {}\n", e.what());
        exit(1);
    }
}