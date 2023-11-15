from math import degrees
import bpy
import bmesh
import json
import bpy.types as T
import mathutils
import sys
from typing import List, Tuple, Dict
from enum import Enum, auto
import json
import os
import struct
from ctypes import *

AKARI_BLENDER_DLL = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), os.environ['AKARI_BLENDER_PATH']))

def check_blender_version():
    # only supports blender 3.6.x and above
    if bpy.app.version[0] < 3:
        print("Blender version must be 3.0 or above")
        sys.exit(1)
    if bpy.app.version[1] < 6:
        print("Blender version must be 3.6 or above")
        sys.exit(1)

check_blender_version()

def dbg(*args, **kwargs):
    print(f"Debug {sys._getframe().f_back.f_lineno}: ", end="")
    print(*args, **kwargs)


def convert_coord_sys_matrix():
    # blender is z-up, right-handed
    # we are y-up, right-handed
    m = mathutils.Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    return m


CONVERT_COORD_SYS_MATRIX = convert_coord_sys_matrix()


def path_join(a, b):
    return os.path.join(a, b).replace("\\", "/")


D = bpy.data
C = bpy.context
depsgraph = C.evaluated_depsgraph_get()
bpy.ops.object.mode_set(mode="OBJECT")
# select one object to activate edit mode
for obj in C.scene.objects:
    first_obj = C.scene.objects[0]
    if first_obj.type == "MESH":
        first_obj.select_set(True)
        C.view_layer.objects.active = first_obj
        break

bpy.ops.object.select_all(action="DESELECT")

argv = sys.argv
print(argv)
argv_offset = argv.index("--") + 2
BLEND_FILE_ABS_PATH = argv[argv_offset - 1]
argv = argv[argv_offset:]
force_export = "--force" in argv or "-f" in argv
update_mesh_only = "--update-mesh-only" in argv or "-u" in argv
save_modified_blend = "--save-modified-blend" in argv or "-s" in argv
BLEND_FILE_DIR = os.path.dirname(BLEND_FILE_ABS_PATH)

OUT_DIR = argv[0]
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
else:
    # check if OUT_DIR is empty
    if len(os.listdir(OUT_DIR)) != 0:
        if not force_export and not update_mesh_only:
            print(f"OUT_DIR `{OUT_DIR}` is not empty")
            print("Use --force or -f to force export")
            sys.exit(1)
        else:
            if force_export:
                print("Force export")
            if update_mesh_only:
                print("Update mesh only")
akr_file = open(os.path.join(OUT_DIR, "scene.json"), "w")
MESH_DIR = path_join(OUT_DIR, "meshes")
TEXTURE_DIR = path_join(OUT_DIR, "textures")
if not os.path.exists(MESH_DIR):
    os.makedirs(MESH_DIR)

if not os.path.exists(TEXTURE_DIR):
    os.makedirs(TEXTURE_DIR)


class UniqueName:
    def __init__(self):
        self.m = dict()
        self.names = dict()

    def contains(self, m):
        assert m is not None
        return m in self.m

    def get(self, m):
        assert m is not None
        if m in self.m:
            return self.m[m]
        name = m.name
        name = name.replace(" ", "_")
        name = name.replace(".", "_")
        name = name.replace("-", "_")

        if name not in self.names:
            self.names[name] = 0
            gened_name = name
        else:
            gened_name = f"{name}_{self.names[name]}"
            self.names[name] += 1
        self.m[m] = gened_name
        dbg(name, gened_name)
        return gened_name


def compute_uv_map(obj):
    # compute uv mapping using smart uv project
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    C.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)


def is_uv_map_valid(obj):
    if len(obj.data.uv_layers) == 0:
        return False
    layer = obj.data.uv_layers[0]
    for uv in layer.uv:
        v = uv.vector
        if v[0] != 0.0 or v[1] != 0.0:
            return True
    return False


VISITED = UniqueName()


def toposort(node_tree):
    nodes = node_tree.nodes
    visited = set()
    order = list()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        order.append(node)
        for input in node.inputs:
            for link in input.links:
                from_node = link.from_node
                dbg("recurse!", node, from_node)
                dfs(from_node)

    sorted = []
    for node in nodes:
        order = []
        dfs(node)
        order = list(reversed(order))
        sorted.extend(order)
    assert len(sorted) == len(nodes)
    return sorted


class NodeType:
    pass


# PrimitiveNodeType = Enum("PrimitiveNodeType", ["Float", "Float3", "RGB", "Spectrum", "BSDF"])
class PrimitiveNodeType(NodeType, Enum):
    FLOAT = auto()
    FLOAT3 = auto()
    RGB = auto()
    SPECTRUM = auto()
    BSDF = auto()


class CompositeNodeType(NodeType):
    def __init__(self, fields) -> None:
        self.fields = fields

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, CompositeNodeType):
            return False
        return self.fields == o.fields


class MaterialExporter:
    socket_to_node_output: Dict[T.NodeSocket, Tuple[T.Node, str]] = dict()
    visited: Dict[T.Node, NodeType]

    def __init__(self) -> None:
        self.visited = dict()
        self.output_node = None
        self.shader_graph = dict()

    def push_node(self, node):
        name = f"$tmp_{len(self.shader_graph)}"
        self.shader_graph[name] = node
        return {"id": name}

    def get_node_input(self, node, key, node_ty, akr_key):
        try:
            s = node.inputs[key]
        except KeyError as e:
            print(f"Node `{node.name}` has no input `{key}`")
            print(f"keys: {node.inputs.keys()}")
            raise e
        connected = len(s.links) != 0
        assert not connected or len(s.links) == 1
        if not connected:
            if (
                isinstance(s, T.NodeSocketFloatFactor)
                or isinstance(s, T.NodeSocketFloat)
                or isinstance(s, T.NodeSocketFloatPercentage)
                or isinstance(s, T.NodeSocketFloatDistance)
                or isinstance(s, T.NodeSocketFloatAngle)
                or isinstance(s, T.NodeSocketFloatUnsigned)
            ):
                assert (
                    node_ty == PrimitiveNodeType.FLOAT
                ), f"Epxected {PrimitiveNodeType.FLOAT} but got {node_ty}"
                return self.push_node({"Float": s.default_value})
            elif isinstance(s, T.NodeSocketColor):
                r = s.default_value[0]
                g = s.default_value[1]
                b = s.default_value[2]
                assert (
                    node_ty == PrimitiveNodeType.RGB
                    or node_ty == PrimitiveNodeType.SPECTRUM
                ), f"Epxected {PrimitiveNodeType.RGB} or {PrimitiveNodeType.SPECTRUM} but got {node_ty}"
                rgb = self.push_node(
                    {"Rgb": {"value": [r, g, b], "colorspace": "srgb"}}
                )
                if node_ty == PrimitiveNodeType.RGB:
                    return rgb
                else:
                    uplift = self.push_node({"SpectralUplift": rgb})
                    return uplift
            else:
                raise RuntimeError(f"Unsupported socket type `{s}`")
        else:
            link = s.links[0]
            from_socket = link.from_socket
            (from_node, from_key) = self.socket_to_node_output[from_socket]
            assert from_socket.node == from_node, f"{from_socket.node} {from_node}"
            has_single_output = len(from_node.outputs) == 1

            input_ty = self.visited[from_node]
            if isinstance(input_ty, PrimitiveNodeType):
                input_node = {"id": VISITED.get(from_node)}
            else:
                input_node = self.push_node(
                    {
                        "ExtractElement": {
                            "node": VISITED.get(from_node),
                            "field": akr_key,
                        }
                    }
                )
            if isinstance(input_ty, PrimitiveNodeType):
                if node_ty == input_ty:
                    pass
                else:
                    NotImplementedError()
            elif isinstance(input_ty, CompositeNodeType):
                input_ty = input_ty.fields[akr_key]
            if node_ty == PrimitiveNodeType.SPECTRUM:
                if input_ty == PrimitiveNodeType.RGB:
                    input_node = self.push_node({"SpectralUplift": input_node})
            else:
                assert node_ty == input_ty, f"{node_ty} {input_ty}"
            return input_node

    def export_node(self, node):
        if node in self.visited:
            return
        name = VISITED.get(node)

        mat = {}
        node_ty = None

        def input(blender_key, akr_key, node_ty):
            mat[akr_key] = self.get_node_input(node, blender_key, node_ty, akr_key)

        if isinstance(node, T.ShaderNodeBsdfPrincipled):
            input("Base Color", "color", PrimitiveNodeType.SPECTRUM)
            input("Roughness", "roughness", PrimitiveNodeType.FLOAT)
            input("Metallic", "metallic", PrimitiveNodeType.FLOAT)
            input("Specular", "specular", PrimitiveNodeType.FLOAT)
            input("Specular Tint", "specular_tint", PrimitiveNodeType.FLOAT)
            input("Emission", "emission", PrimitiveNodeType.SPECTRUM)
            input("Emission Strength", "emission_strength", PrimitiveNodeType.FLOAT)
            input("Clearcoat", "clearcoat", PrimitiveNodeType.FLOAT)
            input("Clearcoat Roughness", "clearcoat_roughness", PrimitiveNodeType.FLOAT)
            input("Transmission", "transmission", PrimitiveNodeType.FLOAT)
            input("IOR", "ior", PrimitiveNodeType.FLOAT)
            mat = {"PrincipledBsdf": mat}
            node_ty = PrimitiveNodeType.BSDF
        elif isinstance(node, T.ShaderNodeOutputMaterial):
            assert self.output_node is None, "Multiple output node"
            input("Surface", "surface", PrimitiveNodeType.BSDF)
            mat = {"OutputSurface": mat}
            self.output_node = name
        elif isinstance(node, T.ShaderNodeBsdfGlass):
            input("Color", "color", PrimitiveNodeType.SPECTRUM)
            input("Roughness", "roughness", PrimitiveNodeType.FLOAT)
            input("IOR", "ior", PrimitiveNodeType.FLOAT)
            mat = {"GlassBsdf": mat}
            node_ty = PrimitiveNodeType.BSDF
        elif isinstance(node, T.ShaderNodeBsdfDiffuse):
            input("Color", "color", PrimitiveNodeType.SPECTRUM)
            mat = {"DiffuseBsdf": mat}
            node_ty = PrimitiveNodeType.BSDF
        elif isinstance(node, T.ShaderNodeEmission):
            input("Color", "color", PrimitiveNodeType.SPECTRUM)
            input("Strength", "strength", PrimitiveNodeType.FLOAT)
            mat = {"Emission": mat}
            node_ty = PrimitiveNodeType.BSDF
        elif isinstance(node, T.ShaderNodeTexImage):
            extension = {
                "REPEAT": "Repeat",
                "EXTEND": "Extend",
                "CLIP": "Clip",
                "MIRROR": "Mirror",
            }[node.extension]
            interpolation = {
                "Closest": "Nearest",
                "Linear": "Linear",
                "Cubic": "Linear",
                "Smart": "Linear",
            }[node.interpolation]
            assert node.image.filepath != ""
            filepath = node.image.filepath
            filepath = filepath.replace("//", "./")
            filepath = filepath.replace("\\", "/")

            filepath = os.path.normpath(os.path.join(BLEND_FILE_DIR, filepath))
            filepath = filepath.replace("\\", "/")
            mat = {
                "TexImage": {
                    "path": filepath,
                    "extension": extension,
                    "interpolation": interpolation,
                    "colorspace": {"Rgb": "srgb"},
                }
            }
            node_ty = PrimitiveNodeType.RGB
        else:
            raise RuntimeError(f"Unsupported node type `{node.type}`")
        self.visited[node] = node_ty
        self.shader_graph[name] = mat

    def export_material(self, mat) -> dict:
        assert mat is not None
        name = VISITED.get(mat)
        print(f"Exporting Material `{mat.name}` -> {name}")
        self.output_node = None
        node_tree = mat.node_tree
        sorted_nodes = toposort(node_tree)
        dbg(sorted_nodes)
        for node in sorted_nodes:
            # if isinstance(node, T.ShaderNodeMixShader):
            #     self.socket_to_node_output[node.inputs[0]] = (node, "fac")
            #     self.socket_to_node_output[node.inputs[1]] = (node, "bsdf1")
            #     self.socket_to_node_output[node.inputs[2]] = (node, "bsdf2")
            # else:
            for key in node.outputs.keys():
                # print(node.outputs[key])
                def rename_key(key):
                    return key.lower().replace(" ", "_")

                self.socket_to_node_output[node.outputs[key]] = (node, rename_key(key))
        for node in sorted_nodes:
            dbg(node)
            self.export_node(node)
        assert self.output_node is not None
        return name


def make_external_buffer(path):
    return {"External": path}


def make_ref(id):
    return {"id": id}


class SceneExporter:
    def __init__(self, scene):
        self.scene = scene
        self.exported_instances = dict()
        self.exported_geometries = dict()
        self.exported_materials = dict()
        self.material_cache = dict()
        self.image_cache = dict()

    def visible_objects(self):
        return [ob for ob in self.scene.objects if not ob.hide_render]

    def export_material(self, m):
        if m in self.material_cache:
            return self.material_cache[m]
        exporter = MaterialExporter()
        out = exporter.export_material(m)
        self.exported_materials[out] = {
            "shader": {
                "nodes": exporter.shader_graph,
                "out": {"id": exporter.output_node},
            }
        }
        self.material_cache[m] = out
        return out

    def export_mesh_data(self, m, name, has_uv):
        bm = bmesh.new()
        bm.from_mesh(m)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(m)
        bm.free()
        m.calc_loop_triangles()
        if has_uv:
            m.calc_tangents()
        else:
            # only compute normals
            m.calc_normals_split()

        V = m.vertices
        F = m.loop_triangles
        print(f"    #V: {len(V)} #F: {len(F)}")
        mesh_out_path = os.path.join(MESH_DIR, f"{name}")

        export_mesh_args = {
            'out_path': mesh_out_path,
            "loop_tri_ptr":m.loop_triangles[0].as_pointer(),
            'vertex_ptr':m.vertices[0].as_pointer(),
            'uv_ptr':m.uv_layers[0].uv[0].as_pointer() if has_uv else 0,
            'mesh_ptr':m.as_pointer(),
            'num_vertices':len(V),
            'num_triangles':len(F),
        }
        AKARI_BLENDER_DLL.export_blender_mesh(json.dumps(export_mesh_args).encode('utf-8'))

        exported_mesh = {}

        exported_mesh["vertices"] = make_external_buffer(
            path_join(MESH_DIR, f"{name}.vert")
        )
        exported_mesh["indices"] = make_external_buffer(
            path_join(MESH_DIR, f"{name}.ind")
        )
        exported_mesh["normals"] = make_external_buffer(
            path_join(MESH_DIR, f"{name}.normal")
        )
        if has_uv:
            exported_mesh["uvs"] = make_external_buffer(
                path_join(MESH_DIR, f"{name}.uv")
            )
            exported_mesh["tangents"] = make_external_buffer(
                path_join(MESH_DIR, f"{name}.tangent")
            )
            exported_mesh["bitangent_signs"] = make_external_buffer(
                path_join(MESH_DIR, f"{name}.bitangent")
            )
        else:
            exported_mesh["uvs"] = None
            exported_mesh["tangents"] = None
            exported_mesh["bitangent_signs"] = None
        self.exported_geometries[name] = {"Mesh": exported_mesh}

    def export_mesh_data_slow(self, m, name, has_uv):
        bm = bmesh.new()
        bm.from_mesh(m)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        bm.to_mesh(m)
        bm.free()
        m.calc_loop_triangles()
        if has_uv:
            m.calc_tangents()
        else:
            # only compute normals
            m.calc_normals_split()
        V = m.vertices
        F = m.loop_triangles
        print(f"    #V: {len(V)} #F: {len(F)}")
        vert_buffer = open(os.path.join(MESH_DIR, f"{name}.vert"), "wb")
        ind_buffer = open(os.path.join(MESH_DIR, f"{name}.ind"), "wb")
        if has_uv:
            uv_buffer = open(os.path.join(MESH_DIR, f"{name}.uv"), "wb")
            tangent_buffer = open(os.path.join(MESH_DIR, f"{name}.tangent"), "wb")
            bitangent_buffer = open(os.path.join(MESH_DIR, f"{name}.bitangent"), "wb")
        normal_buffer = open(os.path.join(MESH_DIR, f"{name}.normal"), "wb")
        bit_counter = 0
        packed_bits = 0
        vert_buffer.write(struct.pack("Q", len(V)))
        ind_buffer.write(struct.pack("Q", len(F)))
        normal_buffer.write(struct.pack("Q", len(F) * 3))
        if has_uv:
            uv_buffer.write(struct.pack("Q", len(F) * 3))
            tangent_buffer.write(struct.pack("Q", len(F) * 3))
            bitangent_buffer.write(struct.pack("Q", (len(F) * 3 + 31) // 32))
        for v in V:
            vert_buffer.write(struct.pack("fff", *v.co))
        for f in F:
            indices = f.vertices
            ind_buffer.write(struct.pack("III", indices[0], indices[1], indices[2]))
            normals = f.split_normals
            for n in normals:
                normal_buffer.write(struct.pack("fff", *n))
            for loop_index in f.loops:
                if has_uv:
                    uv = m.uv_layers[0].uv[loop_index].vector
                    uv_buffer.write(struct.pack("ff", uv[0], uv[1]))
                    tangent = m.loops[loop_index].tangent
                    tangent_buffer.write(struct.pack("fff", *tangent))
                    bitangent_sign = m.loops[loop_index].bitangent_sign
                    if bitangent_sign < 0:
                        bitangent_sign = 1
                    else:
                        bitangent_sign = 0
                    packed_bits |= bitangent_sign << bit_counter
                    bit_counter += 1
                    if bit_counter == 32:
                        bitangent_buffer.write(struct.pack("I", packed_bits))
                        packed_bits = 0
                        bit_counter = 0
        if bit_counter != 0 and has_uv:
            bitangent_buffer.write(struct.pack("I", packed_bits))
        vert_buffer.close()
        ind_buffer.close()
        if has_uv:
            uv_buffer.close()
            tangent_buffer.close()
            bitangent_buffer.close()
        normal_buffer.close()

        exported_mesh = {}

        exported_mesh["vertices"] = make_external_buffer(
            path_join(MESH_DIR, f"{name}.vert")
        )
        exported_mesh["indices"] = make_external_buffer(
            path_join(MESH_DIR, f"{name}.ind")
        )
        exported_mesh["normals"] = make_external_buffer(
            path_join(MESH_DIR, f"{name}.normal")
        )
        if has_uv:
            exported_mesh["uvs"] = make_external_buffer(
                path_join(MESH_DIR, f"{name}.uv")
            )
            exported_mesh["tangents"] = make_external_buffer(
                path_join(MESH_DIR, f"{name}.tangent")
            )
        else:
            exported_mesh["uvs"] = None
            exported_mesh["tangents"] = None
            # exported_mesh["bitangent_signs"] = None
        self.exported_geometries[name] = {"Mesh": exported_mesh}

    def convert_matrix_to_list(self, mat):
        l = [[x for x in row] for row in mat.row]
        return l

    def export_mesh(self, obj):  # TODO: support instancing
        if VISITED.contains(obj):
            return
        name = VISITED.get(obj)
        print(f"Exporting Mesh `{obj.name}` -> {name}")
        # print(obj.data)
        # print(obj.matrix_local)
        print(f"World matrix:")
        print(obj.matrix_world)
        # print(self.convert_matrix_to_list(obj.matrix_world))
        # print(self.transfrom_from_object(obj))
        m = obj.data
        assert (
            len(m.uv_layers) <= 1
        ), f"Only one uv layer is supported but found {len(m.uv_layers)}"
        has_uv = len(m.uv_layers) == 1
        if has_uv:
            if not is_uv_map_valid(obj):
                print(f"Mesh `{obj.name}` has invalid uv map")
                has_uv = False
        if not has_uv:
            print(f"Mesh `{obj.name}` has no uv map")
            print(f"Try to compute uv map for `{obj.name}`")
            try:
                compute_uv_map(obj)
                has_uv = True
            except Exception as e:
                print(f"Failed to compute uv map for `{obj.name}`")
                print(f"Reason: {e}")
                print("Continue without uv map")
        else:
            print(f"Mesh `{obj.name}` has uv map")
        eval_obj = obj.evaluated_get(depsgraph)
        m = eval_obj.to_mesh()
        self.export_mesh_data(m, f"{name}_mesh", has_uv)
        assert (
            len(obj.data.materials) == 1
        ), f"Mesh `{obj.name}` has {len(obj.data.materials)} materials"
        mat = self.export_material(obj.data.materials[0])
        matrix_world = CONVERT_COORD_SYS_MATRIX @ obj.matrix_world
        print(matrix_world)
        instance = {
            "geometry": make_ref(f"{name}_mesh"),
            "material": make_ref(f"{mat}"),
            "transform": {"Matrix": self.convert_matrix_to_list(matrix_world)},
        }
        self.exported_instances[name] = instance

    def transfrom_from_object(self, b_object):
        translate = b_object.location
        rotate = b_object.rotation_euler
        scale = b_object.scale
        return (translate, rotate, scale)

    def export_camera(self):
        camera = self.scene.camera
        fov = degrees(camera.data.angle)
        name = VISITED.get(camera)
        print(f"Exporting Camera `{camera.name}`")
        print(type(camera))
        print(type(camera.data))
        exported_camera = {}
        trs = self.transfrom_from_object(camera)
        dof = camera.data.dof
        render_settings = self.scene.render
        res_x = render_settings.resolution_x
        res_y = render_settings.resolution_y
        exported_camera["transform"] = {
            "TRS": {
                "translation": [x for x in trs[0]],
                "rotation": [x for x in trs[1]],
                "scale": [x for x in trs[2]],
                "coordinate_system": "Blender",
            }
        }
        exported_camera["fov"] = fov
        if dof is not None:
            exported_camera["focal_distance"] = dof.focus_distance
            exported_camera["fstop"] = dof.aperture_fstop
        exported_camera["sensor_width"] = res_x
        exported_camera["sensor_height"] = res_y
        return {"Perspective": exported_camera}

    def export(self):
        name = VISITED.get(self.scene)
        camera = self.export_camera()
        print(f"Exporting Scene `{self.scene.name}` -> {name}")
        # seperate meshes by material
        print("Seperating meshes by material")
        for obj in self.visible_objects():
            if obj.type == "MESH":
                bpy.ops.object.mode_set(mode="OBJECT")
                bpy.ops.object.select_all(action="DESELECT")
                obj.select_set(True)
                bpy.ops.object.mode_set(mode="EDIT")
                # Seperate by material
                bpy.ops.mesh.separate(type="MATERIAL")
                bpy.ops.object.mode_set(mode="OBJECT")
                obj.select_set(False)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in self.visible_objects():
            if obj.type == "MESH":
                self.export_mesh(obj)
        scene = {
            "camera": camera,
            "instances": self.exported_instances,
            "lights": {},
            "geometries": self.exported_geometries,
            "materials": self.exported_materials,
        }
        json.dump(scene, akr_file, indent=4)


def export_scene(scene):
    exporter = SceneExporter(scene)
    exporter.export()


export_scene(C.scene)
akr_file.close()
print("Export finished")
if save_modified_blend:
    print("Saving modified blend file")
    bpy.ops.wm.save_mainfile()
# for debugging
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "modified.blend"))
