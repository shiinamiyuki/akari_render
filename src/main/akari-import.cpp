// Copyright 2020 shiinamiyuki
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assimp/Importer.hpp> // C++ importer interface
#include <assimp/scene.h>      // Output data structure
#include <assimp/postprocess.h>
#include <fstream>
#include <akari/util.h>
#include <akari/scenegraph.h>
#include <akari/serial.h>
using namespace akari::scene;
using akari::Spectrum;
P<SceneGraph> import(const std::string &file) {
    auto create_vec = [&](aiVector3D v) -> glm::vec3 { return glm::vec3(v[0], v[1], v[2]); };
    auto color4_to_spectrum = [&](aiColor4D v) -> Spectrum { return Spectrum(v[0], v[1], v[2]); };
    Assimp::Importer importer;
    importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 20);
    const aiScene *ai_scene =
        importer.ReadFile(file, aiProcess_CalcTangentSpace | aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                                    aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
    P<SceneGraph> scene(new SceneGraph());
    std::vector<P<Material>> materials((size_t)ai_scene->mNumMaterials);
    std::vector<P<Mesh>> meshes((size_t)ai_scene->mNumMeshes);
    for (uint32_t i = 0; i < ai_scene->mNumMaterials; i++) {
        const aiMaterial *ai_mat = ai_scene->mMaterials[i];
        aiColor4D diffuse_color, emission_color;
        aiGetMaterialColor(ai_mat, AI_MATKEY_COLOR_DIFFUSE, &diffuse_color);
        aiGetMaterialColor(ai_mat, AI_MATKEY_COLOR_EMISSIVE, &emission_color);
        auto mat = P<Material>(new Material());
        mat->color = P<Texture>(new Texture(color4_to_spectrum(diffuse_color)));
        mat->emission = P<Texture>(new Texture(color4_to_spectrum(emission_color)));
        materials[i] = mat;
    }
    for (uint32_t i = 0; i < ai_scene->mNumMeshes; i++) {
        const aiMesh *ai_mesh = ai_scene->mMeshes[i];
        if (!(ai_mesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE)) {
            continue;
        }
        auto mesh = std::make_shared<Mesh>();
        mesh->name = ai_mesh->mName.C_Str();
        mesh->indices.resize(ai_mesh->mNumFaces);
        for (uint32_t t = 0; t < ai_mesh->mNumFaces; t++) {
            AKR_ASSERT(ai_mesh->mFaces[t].mNumIndices == 3);
            auto f = ai_mesh->mFaces[t].mIndices;
            mesh->indices[t] = glm::ivec3(f[0], f[1], f[2]);
        }
        mesh->vertices.resize(ai_mesh->mNumVertices);
        for (uint32_t vt = 0; vt < ai_mesh->mNumVertices; vt++) {
            auto v = ai_mesh->mVertices[vt];
            mesh->vertices[vt] = glm::vec3(v[0], v[1], v[2]);
        }
        if (!ai_mesh->HasNormals()) {
            fprintf(stderr, "normal is missing!\n");
            exit(1);
        }
        mesh->normals.resize(ai_mesh->mNumVertices);
        for (uint32_t nt = 0; nt < ai_mesh->mNumVertices; nt++) {
            auto n = ai_mesh->mNormals[nt];
            mesh->normals[nt] = glm::vec3(n[0], n[1], n[2]);
        }

        if (ai_mesh->HasTextureCoords(0)) {
            mesh->texcoords.resize(ai_mesh->mNumVertices);
            for (uint32_t t = 0; t < ai_mesh->mNumVertices; t++) {
                auto tc = ai_mesh->mTextureCoords[0][t];
                mesh->texcoords[t] = glm::vec2(tc[0], tc[1]);
            }
        }
        meshes[i] = mesh;
    }

    auto create_transform = [&](aiMatrix4x4 m) -> akari::TRSTransform {
        aiVector3D t, r, s;
        m.Decompose(s, r, t);
        return akari::TRSTransform(create_vec(t), create_vec(r), create_vec(s));
    };
    auto create_node = [&](const aiNode *ai_node, auto &&self) -> P<Node> {
        P<Node> node(new Node());
        for (uint32_t i = 0; i < ai_node->mNumChildren; i++) {
            node->children.emplace_back(self(ai_node->mChildren[i], self));
        }
        for (uint32_t i = 0; i < ai_node->mNumMeshes; i++) {
            P<Instance> instance(new Instance());
            instance->name = meshes[ai_node->mMeshes[i]]->name;
            instance->transform = create_transform(ai_node->mTransformation);
            instance->mesh = meshes[ai_node->mMeshes[i]];
            instance->material = materials[ai_scene->mMeshes[ai_node->mMeshes[i]]->mMaterialIndex];
            node->instances.emplace_back(instance);
        }
        return node;
    };
    scene->camera.reset(new PerspectiveCamera());
    scene->meshes = meshes;
    scene->root = create_node(ai_scene->mRootNode, create_node);
    printf("imported %zd meshes\n", scene->meshes.size());
    return scene;
}
void save(const P<SceneGraph> &scene, std::string dir) {
    using namespace akari;
    for (size_t i = 0; i < scene->meshes.size(); i++) {
        auto file = fs::path(dir) / std::string("mesh" + std::to_string(i));
        auto &mesh = scene->meshes[i];
        mesh->path = file.string();
        mesh->save_to_file(mesh->path);
    }
}
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: akari-import file scene-file output-dir\n");
        exit(1);
    }
    std::string file(argv[1]);
    std::string scene_file(argv[2]);
    std::string out_dir(argv[3]);
    using namespace akari;
    if (fs::exists(out_dir) && !fs::is_empty(out_dir)) {
        fprintf(stderr, "folder %s already exists and is not empty\n", out_dir.c_str());
        exit(1);
    }
    if (!fs::exists(out_dir)) {
        fs::create_directory(out_dir);
    }
    auto scene = import(file);
    save(scene, out_dir);
    {
        std::ofstream os(scene_file);
        cereal::JSONOutputArchive ar(os);
        ar(scene);
    }
    printf("done importing\n");
}