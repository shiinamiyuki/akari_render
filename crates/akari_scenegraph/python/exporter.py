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
import time

AKARI_API = cdll.LoadLibrary(os.environ["AKARI_API_PATH"])
py_akari_import = AKARI_API.py_akari_import
py_akari_import.restype = c_ulonglong
py_akari_free_string = AKARI_API.py_akari_free_string
py_akari_free_string.argtypes = (c_ulonglong,)


def invoke_akari_api(args):
    # print(args)
    args = json.dumps(args, indent=1).encode("utf-8")
    ptr = py_akari_import(args, c_ulonglong(len(args)))
    ret = string_at(ptr).decode("utf-8")
    ret = json.loads(ret)
    AKARI_API.py_akari_free_string(ptr)
    return ret


def check_blender_version():
    # only supports blender 4.0
    if bpy.app.version[0] != 4:
        print("Only supports blender 4.0")
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
akr_file = os.path.join(OUT_DIR, "scene.json")
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
            generated_name = name
        else:
            generated_name = f"{name}_{self.names[name]}"
            self.names[name] += 1
        self.m[m] = generated_name
        dbg(name, generated_name)
        return generated_name


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
                ), f"Expected {PrimitiveNodeType.FLOAT} but got {node_ty}"
                return self.push_node({"value": s.default_value, 'type': 'float'})
            elif isinstance(s, T.NodeSocketColor):
                r = s.default_value[0]
                g = s.default_value[1]
                b = s.default_value[2]
                assert (
                        node_ty == PrimitiveNodeType.RGB
                        or node_ty == PrimitiveNodeType.SPECTRUM
                ), f"Expected {PrimitiveNodeType.RGB} or {PrimitiveNodeType.SPECTRUM} but got {node_ty}"
                rgb = self.push_node(
                    {"type": "rgb", "value": [r, g, b], "colorspace": "srgb"}
                )
                if node_ty == PrimitiveNodeType.RGB:
                    return rgb
                else:
                    uplift = self.push_node({"rgb": rgb, 'type': 'spectral_uplift'})
                    return uplift
            elif isinstance(s, T.NodeSocketVector):
                x = s.default_value[0]
                y = s.default_value[1]
                z = s.default_value[2]
                assert (
                        node_ty == PrimitiveNodeType.FLOAT3
                ), f"Expected {PrimitiveNodeType.FLOAT3} but got {node_ty}"
                return self.push_node({"type": "float3", "value": [x, y, z]})
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
                        "type": "extract",
                        "node": VISITED.get(from_node),
                        "field": akr_key,
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
            input("Base Color", "base_color", PrimitiveNodeType.SPECTRUM)
            input("Roughness", "roughness", PrimitiveNodeType.FLOAT)
            input("Metallic", "metallic", PrimitiveNodeType.FLOAT)
            input("Specular IOR Level", "specular_ior_level", PrimitiveNodeType.FLOAT)
            input("Specular Tint", "specular_tint", PrimitiveNodeType.SPECTRUM)
            input("Emission Color", "emission_color", PrimitiveNodeType.SPECTRUM)
            input("Emission Strength", "emission_strength", PrimitiveNodeType.FLOAT)
            input("Coat Weight", "coat_weight", PrimitiveNodeType.FLOAT)
            input("Coat Roughness", "coat_roughness", PrimitiveNodeType.FLOAT)
            input('Coat Normal', 'coat_normal', PrimitiveNodeType.FLOAT3)
            input('Coat IOR', 'coat_ior', PrimitiveNodeType.FLOAT)
            input('Coat Tint', 'coat_tint', PrimitiveNodeType.SPECTRUM)
            input("Transmission Weight", "transmission_weight", PrimitiveNodeType.FLOAT)
            input("IOR", "ior", PrimitiveNodeType.FLOAT)
            input('Sheen Weight', 'sheen_weight', PrimitiveNodeType.FLOAT)
            input('Sheen Tint', 'sheen_tint', PrimitiveNodeType.SPECTRUM)
            input('Subsurface Weight', 'subsurface_weight', PrimitiveNodeType.FLOAT)
            input('Subsurface Radius', 'subsurface_radius', PrimitiveNodeType.FLOAT3)
            input('Subsurface Scale', 'subsurface_scale', PrimitiveNodeType.FLOAT)
            input('Subsurface IOR', 'subsurface_ior', PrimitiveNodeType.FLOAT)
            input('Subsurface Anisotropy', 'subsurface_anisotropy', PrimitiveNodeType.FLOAT)
            input('Anisotropic Rotation', 'anisotropic_rotation', PrimitiveNodeType.FLOAT)
            input('Anisotropic', 'anisotropic', PrimitiveNodeType.FLOAT)
            input('Alpha', 'alpha', PrimitiveNodeType.FLOAT)
            input('Normal', 'normal', PrimitiveNodeType.FLOAT3)
            input('Tangent', 'tangent', PrimitiveNodeType.FLOAT3)
            mat['type'] = 'principled'
            mat['preference'] = 'mix'
            node_ty = PrimitiveNodeType.BSDF
        elif isinstance(node, T.ShaderNodeOutputMaterial):
            assert self.output_node is None, "Multiple output node"
            input("Surface", "node", PrimitiveNodeType.BSDF)
            mat['type'] = 'output'
            self.output_node = name
        elif isinstance(node, T.ShaderNodeBsdfGlass):
            input("Color", "color", PrimitiveNodeType.SPECTRUM)
            input("Roughness", "roughness", PrimitiveNodeType.FLOAT)
            input("IOR", "ior", PrimitiveNodeType.FLOAT)
            mat['type'] = 'glass'
            node_ty = PrimitiveNodeType.BSDF
        elif isinstance(node, T.ShaderNodeBsdfDiffuse):
            input("Color", "color", PrimitiveNodeType.SPECTRUM)
            mat['type'] = 'diffuse'
            node_ty = PrimitiveNodeType.BSDF
        elif isinstance(node, T.ShaderNodeEmission):
            input("Color", "color", PrimitiveNodeType.SPECTRUM)
            input("Strength", "strength", PrimitiveNodeType.FLOAT)
            mat['type'] = 'emission'
            node_ty = PrimitiveNodeType.BSDF
        elif isinstance(node, T.ShaderNodeTexImage):
            extension = {
                "REPEAT": "repeat",
                "EXTEND": "extend",
                "CLIP": "clip",
                "MIRROR": "mirror",
            }[node.extension]
            interpolation = {
                "Closest": "nearest",
                "Linear": "linear",
                "Cubic": "linear",
                "Smart": "linear",
            }[node.interpolation]
            assert node.image.filepath != ""
            filepath = node.image.filepath
            filepath = filepath.replace("//", "./")
            filepath = filepath.replace("\\", "/")

            filepath = os.path.normpath(os.path.join(BLEND_FILE_DIR, filepath))
            filepath = filepath.replace("\\", "/")
            mat = {
                'type': 'image',
                "image": {
                    "path": filepath,
                    "extension": extension,
                    "interpolation": interpolation,
                    "colorspace": "srgb",
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
        self.material_cache = dict()
        self.image_cache = dict()

    def visible_objects(self):
        return [ob for ob in self.scene.objects if not ob.hide_render]

    def export_material(self, m):
        if m in self.material_cache:
            return self.material_cache[m]
        exporter = MaterialExporter()
        out = exporter.export_material(m)
        out = invoke_akari_api({
            "ImportMaterial": {
                "name": out,
                "mat": {
                    "shader": {
                        "nodes": exporter.shader_graph,
                        "output": {"id": exporter.output_node},
                        "kind": "surface"
                    }
                }
            }
        })['Material']['value']
        self.material_cache[m] = out
        return out

    def export_mesh_data(self, m, name, has_uv):
        # bm = bmesh.new()
        # bm.from_mesh(m)
        # bmesh.ops.triangulate(bm, faces=bm.faces[:])
        # bm.to_mesh(m)
        # bm.free()
        m.calc_loop_triangles()

        V = m.vertices
        F = m.loop_triangles
        print(f"    #V: {len(V)} #F: {len(F)}")

        export_mesh_args = {
            "name": name,
            "loop_tri_ptr": m.loop_triangles[0].as_pointer(),
            "vertex_ptr": m.vertices[0].as_pointer(),
            "uv_ptr": m.uv_layers[0].uv[0].as_pointer() if has_uv else 0,
            "mesh_ptr": m.as_pointer(),
            "num_vertices": len(V),
            "num_triangles": len(F),
            'has_multi_materials': len(m.materials) > 1,
        }
        exported_mesh = invoke_akari_api(
            {
                "ImportMesh": {
                    "args": export_mesh_args,
                }
            }
        )

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
        eval_obj = obj.evaluated_get(depsgraph)
        m = eval_obj.to_mesh()
        self.export_mesh_data(m, f"{name}_mesh", has_uv)
        assert (
                len(obj.data.materials) == 1
        ), f"Mesh `{obj.name}` has {len(obj.data.materials)} materials"
        mat = [self.export_material(m) for m in obj.data.materials]
        matrix_world = CONVERT_COORD_SYS_MATRIX @ obj.matrix_world
        print(matrix_world)
        instance = {
            "geometry": make_ref(f"{name}_mesh"),
            "materials": mat,
            "transform": {"type": "matrix", "data": self.convert_matrix_to_list(matrix_world)},
        }
        invoke_akari_api({
            'ImportInstance': {
                'name': name,
                'instance': instance,
            }
        })

    def transfrom_from_object(self, b_object):
        translate = b_object.location
        rotate = b_object.rotation_euler
        scale = b_object.scale
        return translate, rotate, scale

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
            "data": {
                "translation": [x for x in trs[0]],
                "rotation": [x for x in trs[1]],
                "scale": [x for x in trs[2]],
                "coordinate_system": "Blender",
            },
            "type": "trs"
        }
        exported_camera["fov"] = fov
        if dof is not None:
            exported_camera["focal_distance"] = dof.focus_distance
            exported_camera["fstop"] = dof.aperture_fstop
        exported_camera["sensor_width"] = res_x
        exported_camera["sensor_height"] = res_y
        return {"data": exported_camera, "type": "perspective"}

    def export(self):
        name = VISITED.get(self.scene)
        camera = self.export_camera()
        invoke_akari_api({
            'ImportCamera': {
                'camera': camera,
            }
        })
        print(f"Exporting Scene `{self.scene.name}` -> {name}")

        for obj in self.visible_objects():
            if obj.type == "MESH":
                self.export_mesh(obj)


def export_scene(scene):
    invoke_akari_api({
        'Init': {
            'name': scene.name,
        }
    })
    exporter = SceneExporter(scene)
    exporter.export()
    invoke_akari_api({
        'WriteScene': {
            'path': akr_file,
            'compact': True
        }
    })


export_scene(C.scene)
print("Export finished")
if save_modified_blend:
    print("Saving modified blend file")
    bpy.ops.wm.save_mainfile()
# for debugging
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "modified.blend"))
