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
argv_offset = argv.index("--") + 1
argv = argv[argv_offset:]
force_export = "--force" in argv or "-f" in argv
update_mesh_only = "--update-mesh-only" in argv or "-u" in argv
save_modified_blend = "--save-modified-blend" in argv or "-s" in argv


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
if not os.path.exists(MESH_DIR):
    os.makedirs(MESH_DIR)


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


NodeType = Enum("NodeType", ["Float", "Float3", "RGB", "Spectrum", "BSDF"])


class MaterialExporter:
    socket_to_node_output: Dict[T.NodeSocket, Tuple[T.Node, str]] = dict()
    visited: Dict[T.Node, Tuple[dict, NodeType]]

    def __init__(self) -> None:
        self.visited = set()
        self.output_node = None
        self.shader_graph = dict()

    def push_node(self, node):
        name = f"$tmp_{len(self.shader_graph)}"
        self.shader_graph[name] = node
        return {"id": name}

    def get_node_input(self, node, key, node_ty):
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
                    node_ty == NodeType.Float
                ), f"Epxected {NodeType.Float} but got {node_ty}"
                return self.push_node({"Float": s.default_value})
            elif isinstance(s, T.NodeSocketColor):
                r = s.default_value[0]
                g = s.default_value[1]
                b = s.default_value[2]
                assert (
                    node_ty == NodeType.RGB or node_ty == NodeType.Spectrum
                ), f"Epxected {NodeType.RGB} or {NodeType.Spectrum} but got {node_ty}"
                rgb = self.push_node({"Rgb": {'value':[r, g, b], 'colorspace':'srgb'}})
                if node_ty == NodeType.RGB:
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
            if has_single_output:
                return {"id": VISITED.get(from_node)}
            return self.push_node(
                {
                    "ExtractElement": {
                        "node": VISITED.get(from_node),
                        "field": from_key,
                    }
                }
            )

    def export_node(self, node):
        if node in self.visited:
            return
        name = VISITED.get(node)
        self.visited.add(node)
        

        mat = {}

        def input(blender_key, akr_key, node_ty):
            mat[akr_key] = self.get_node_input(node, blender_key, node_ty)

        if isinstance(node, T.ShaderNodeBsdfPrincipled):
            input("Base Color", "color", NodeType.Spectrum)
            input("Roughness", "roughness", NodeType.Float)
            input("Metallic", "metallic", NodeType.Float)
            input("Specular", "specular", NodeType.Float)
            input("Emission", "emission", NodeType.Spectrum)
            input('Emission Strength', 'emission_strength', NodeType.Float)
            input("Clearcoat", "clearcoat", NodeType.Float)
            input("Clearcoat Roughness", "clearcoat_roughness", NodeType.Float)
            input("Transmission", "transmission", NodeType.Float)
            input("IOR", "ior", NodeType.Float)
            mat = {"PrincipledBsdf": mat}
        elif isinstance(node, T.ShaderNodeOutputMaterial):
            assert self.output_node is None, "Multiple output node"
            input("Surface", "surface", NodeType.BSDF)
            mat = {'OutputSurface': mat}
            self.output_node = name
        elif isinstance(node, T.ShaderNodeBsdfGlass):
            input("Color", "color")
            input("Roughness", "roughness")
            input("IOR", "ior")
            mat = {"GlassBsdf": mat}
        elif isinstance(node, T.ShaderNodeBsdfDiffuse):
            input("Color", "color")
            mat = {"DiffuseBsdf": mat}
            
        elif isinstance(node, T.ShaderNodeEmission):
            input("Color", "color")
            input("Strength", "strength")
            mat = {"Emission": mat}
        elif isinstance(node, T.ShaderNodeTexImage):
            raise NotImplementedError()
            # extension = {
            #     "REPEAT": "Repeat",
            #     "EXTEND": "Extend",
            #     "CLIP": "Clip",
            #     "MIRROR": "Mirror",
            # }[node.extension]
            # interpolation = {
            #     "Closest": "Nearest",
            #     "Linear": "Linear",
            #     "Cubic": "Linear",
            #     "Smart": "Linear",
            # }[node.interpolation]
        else:
            raise RuntimeError(f"Unsupported node type `{node.type}`")
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
        self.exported_images = dict()
        self.exported_geometries = dict()
        self.exported_materials = dict()
        self.material_cache  = dict()

    def visible_objects(self):
        return [ob for ob in self.scene.objects if not ob.hide_render]

    def export_material(self, m):
        if m in self.material_cache:
            return self.material_cache[m]
        exporter = MaterialExporter()
        out = exporter.export_material(m)
        self.exported_materials[out] = {'shader':{'nodes':exporter.shader_graph, 'out':{'id':exporter.output_node}}}
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
            exported_mesh["bitangent_signs"] = make_external_buffer(
                path_join(MESH_DIR, f"{name}.bitangent")
            )
        else:
            exported_mesh["uvs"] = None
            exported_mesh["tangents"] = None
            exported_mesh["bitangent_signs"] = None
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
                "coordinate_system": "Blender"
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
            "images": self.exported_images,
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
