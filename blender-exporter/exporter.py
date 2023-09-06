from math import degrees
import bpy
import bmesh
import json
import bpy.types as T
import mathutils
import sys
from typing import List, Tuple, Dict
from enum import Enum, auto
import os
import struct
def dbg(*args, **kwargs):
    print(f"Debug {sys._getframe().f_back.f_lineno}: ", end="")
    print(*args, **kwargs)

def convert_coord_sys_matrix():
    # blender is z-up, right-handed
    # we are y-up, right-handed
    m = mathutils.Matrix([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0,-1, 0, 0],
        [0, 0, 0, 1]
    ])
    return m
CONVERT_COORD_SYS_MATRIX = convert_coord_sys_matrix()
def path_join(a, b):
    return os.path.join(a, b).replace("\\", "/")

D = bpy.data
C = bpy.context
depsgraph = C.evaluated_depsgraph_get()
bpy.ops.object.mode_set(mode='OBJECT')
# select one object to activate edit mode
for obj in C.scene.objects:
    first_obj = C.scene.objects[0]
    if first_obj.type == 'MESH':
        first_obj.select_set(True)
        C.view_layer.objects.active = first_obj
        break

bpy.ops.object.select_all(action='DESELECT')

argv = sys.argv
print(argv)
argv_offset = argv.index("--") + 1
argv = argv[argv_offset:]
force_export ='--force' in argv or '-f' in argv
update_mesh_only = '--update-mesh-only' in argv or '-u' in argv
save_modified_blend = '--save-modified-blend' in argv or '-s' in argv


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
akr_file = open(os.path.join(OUT_DIR, "scene.akr"), "w")
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
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    C.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.mode_set(mode='OBJECT')
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
                dbg('recurse!', node, from_node)
                dfs(from_node)
    sorted = []
    for node in nodes:
        order = []
        dfs(node)
        order = list(reversed(order))
        sorted.extend(order)
    assert len(sorted) == len(nodes)
    return sorted
class MaterialExporter:
    socket_to_node_output: Dict[T.NodeSocket, Tuple[T.Node, str]] = dict()
    def __init__(self) -> None:
        self.visited = set()
        self.output_node = None
        self.exported_materials = dict()
    
    def get_node_input(self, node, key):
        try:
            s = node.inputs[key]
        except KeyError as e:
            print(f"Node `{node.name}` has no input `{key}`")
            print(f'keys: {node.inputs.keys()}')
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
                return f'{s.default_value}'
            elif isinstance(s, T.NodeSocketColor):
                r = s.default_value[0]
                g = s.default_value[1]
                b = s.default_value[2]
                return f'SpectralUplift[rgb=RGB[r={r}, g={g}, b={b}, colorspace=ColorSpace::SRGB]]'
            else:
                raise RuntimeError(f"Unsupported socket type `{s}`")
        else:
            link = s.links[0]
            from_socket = link.from_socket
            (from_node, from_key) = self.socket_to_node_output[from_socket]
            assert from_socket.node == from_node, f"{from_socket.node} {from_node}"
            return f'${VISITED.get(from_node)}.{from_key}'

    def export_node(self, node):
        if node in self.visited:
            return
        name = VISITED.get(node)
        self.visited.add(node)
        print(f'${name} = ', end="",file=akr_file)
        cnt = 0
        def input(blender_key, akr_key):
            nonlocal cnt
            if cnt != 0:
                print(",\n",file=akr_file, end="")
            print(f'    {akr_key}={self.get_node_input(node, blender_key)}', end="",file=akr_file)
            cnt += 1
        if isinstance(node, T.ShaderNodeBsdfPrincipled):
            print(f'PrincipledBsdf[',file=akr_file)
            input("Base Color", 'color')
            input('Roughness', 'roughness')
            input('Metallic', 'metallic')
            input('Specular', 'specular')
            input('Emission', 'emission')
            input('Clearcoat', 'clearcoat')
            input('Clearcoat Roughness', 'clearcoat_roughness')
            input('Transmission', 'transmission')
            input('IOR', 'ior')
        elif isinstance(node, T.ShaderNodeOutputMaterial):
            print(f'MaterialOutput[',file=akr_file)
            input("Surface", 'surface')
            self.output_node = node
        elif isinstance(node, T.ShaderNodeBsdfGlass):
            print(f'GlassBsdf[',file=akr_file)
            input("Color", 'color')
            input('Roughness', 'roughness')
            input('IOR', 'ior')
        elif isinstance(node, T.ShaderNodeEmission):
            print(f'Emission[',file=akr_file)
            input("Color", 'color')
            input('Strength', 'strength')
        elif isinstance(node, T.ShaderNodeTexImage):
            extension = {
                'REPEAT': 'Repeat',
                'EXTEND': 'Extend',
                'CLIP': 'Clip',
                'MIRROR': 'Mirror',
            }[node.extension]
            interpolation = {
                'Closest': 'Nearest',
                'Linear': 'Linear',
                'Cubic': 'Linear',
                'Smart': 'Linear',
            }[node.interpolation]
        else:
            raise RuntimeError(f"Unsupported node type `{node.type}`")
        print("\n]",file=akr_file)

    def export_material(self, mat):
        assert mat is not None
        if mat in self.exported_materials:
            return self.exported_materials[mat]
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
        out = VISITED.get(self.output_node)
        self.exported_materials[mat] = out
        return out

class SceneExporter:
    def __init__(self, scene):
        self.scene = scene
        self.exported_instances = list()
        self.mat_exporter = MaterialExporter()

    def visible_objects(self):
        return [ob for ob in self.scene.objects if not ob.hide_render]

    def export_material(self, m):
        return self.mat_exporter.export_material(m)
    
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
        vert_buffer.write(struct.pack('Q', len(V)))
        ind_buffer.write(struct.pack('Q', len(F)))
        normal_buffer.write(struct.pack('Q', len(F) * 3))
        if has_uv:
            uv_buffer.write(struct.pack('Q', len(F) * 3))
            tangent_buffer.write(struct.pack('Q', len(F) * 3))
            bitangent_buffer.write(struct.pack('Q', (len(F) * 3 + 31) // 32))
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
        print(f'${name}_vert = Buffer[name="vertices", path="{path_join(MESH_DIR, f"{name}.vert")}"]',file=akr_file)
        print(f'${name}_ind = Buffer[name="indices", path="{path_join(MESH_DIR, f"{name}.ind")}"]',file=akr_file)
        print(f'${name}_normal = Buffer[name="normals", path="{path_join(MESH_DIR, f"{name}.normal")}"]',file=akr_file)
        if has_uv:
            print(f'${name}_uv = Buffer[name="uvs", path="{path_join(MESH_DIR, f"{name}.uv")}"]',file=akr_file)
            print(f'${name}_tangent = Buffer[name="tangents", path="{path_join(MESH_DIR, f"{name}.tangent")}"]',file=akr_file)
            print(f'${name}_bitangent = Buffer[name="bitangent_signs", path="{path_join(MESH_DIR, f"{name}.bitangent")}"]',file=akr_file)
        buffers = [f"{name}_vert", f"{name}_ind", f"{name}_normal"]
        if has_uv:
            buffers.extend([f"{name}_uv", f"{name}_tangent", f"{name}_bitangent"])
       
        
        print(f'${name} = Mesh[\n    name="{name}",\n    buffers=[',file=akr_file)
        for i, buffer in enumerate(buffers):
            print(f'        ${buffer}',file=akr_file, end="")
            if i != len(buffers) - 1:
                print(",",file=akr_file)
            else:
                print(file=akr_file)
        print("    ],",file=akr_file)
        print("]",file=akr_file)

    def convert_matrix_to_list(self, mat):
        l = [[x for x in row] for row in mat.row]
        # flatten
        return [x for row in l for x in row]

    def export_mesh(self, obj): # TODO: support instancing
        if VISITED.contains(obj):
            return
        name = VISITED.get(obj)
        print(f"Exporting Mesh `{obj.name}` -> {name}")
        # print(obj.data)
        # print(obj.matrix_local)
        print(f'World matrix:')
        print(obj.matrix_world)
        # print(self.convert_matrix_to_list(obj.matrix_world))
        # print(self.transfrom_from_object(obj))
        m = obj.data
        assert len(m.uv_layers) <= 1, f"Only one uv layer is supported but found {len(m.uv_layers)}"
        has_uv = len(m.uv_layers) == 1
        if has_uv:
            if not is_uv_map_valid(obj):
                print(f"Mesh `{obj.name}` has invalid uv map")
                has_uv = False
        if not has_uv:
            print(f"Mesh `{obj.name}` has no uv map")
            print(f'Try to compute uv map for `{obj.name}`')
            try:
                compute_uv_map(obj)
                has_uv = True
            except Exception as e:
                print(f"Failed to compute uv map for `{obj.name}`")
                print(f'Reason: {e}')
                print("Continue without uv map")
        else:
            print(f"Mesh `{obj.name}` has uv map")
        eval_obj = obj.evaluated_get(depsgraph)
        m = eval_obj.to_mesh()
        self.export_mesh_data(m, f'{name}_mesh', has_uv)
        assert len(obj.data.materials) == 1, f"Mesh `{obj.name}` has {len(obj.data.materials)} materials"
        mat = self.export_material(obj.data.materials[0])
        matrix_world = CONVERT_COORD_SYS_MATRIX @ obj.matrix_world
        print(matrix_world)
        print(f'${name} = Instance[',file=akr_file)
        print(f'    name="{name}",',file=akr_file)
        print(f'    geometry=${name}_mesh,',file=akr_file)
        print(f'    material=${mat},',file=akr_file)
        print(f'    transform=MatrixTransform[matrix=[{",".join(["{:f}".format(x) for x in self.convert_matrix_to_list(matrix_world)])}]],',file=akr_file)
        print(f']',file=akr_file)
        self.exported_instances.append(name)

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
        print(f'${name} = PerspectiveCamera[',file=akr_file)
        trs = self.transfrom_from_object(camera)
        dof = camera.data.dof
        render_settings = self.scene.render
        res_x = render_settings.resolution_x
        res_y = render_settings.resolution_y
        print(f'    transform = TRS[',file=akr_file)
        print(f'        translation=[{",".join(["{:f}".format(x) for x in trs[0]])}],',file=akr_file)
        print(f'        rotation=[{",".join(["{:f}".format(degrees(x)) for x in trs[1]])}],',file=akr_file)
        print(f'        scale=[{",".join(["{:f}".format(x) for x in trs[2]])}],',file=akr_file)
        print(f'        coordinate_system=CoordinateSystem::Blender',file=akr_file)
        print(f'    ],',file=akr_file)
        print(f'    fov={fov},',file=akr_file)
        if dof is not None:
            print(f'    focal_distance={dof.focus_distance},',file=akr_file)
            print(f'    fstop={dof.aperture_fstop},',file=akr_file)
        print(f'    width={res_x},',file=akr_file)
        print(f'    height={res_y},',file=akr_file)
        print(f']',file=akr_file)
        return name

    def export(self):
        name = VISITED.get(self.scene)
        camera = self.export_camera()
        print(f"Exporting Scene `{self.scene.name}` -> {name}")
        # seperate meshes by material
        print("Seperating meshes by material")
        for obj in self.visible_objects():
            if obj.type == 'MESH':
                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                bpy.ops.object.mode_set(mode='EDIT')
                # Seperate by material
                bpy.ops.mesh.separate(type='MATERIAL')
                bpy.ops.object.mode_set(mode='OBJECT')
                obj.select_set(False)
        bpy.ops.object.select_all(action='DESELECT')
        for obj in self.visible_objects():
            if obj.type == 'MESH':
                self.export_mesh(obj)
        print(f'${name} = Scene[\n    instances=[',file=akr_file)
        for i, m in enumerate(self.exported_instances):
            print(f'    ${m}',file=akr_file, end="")
            if i != len(self.exported_instances) - 1:
                print(",",file=akr_file)
            else:
                print(file=akr_file)
        print("    ],",file=akr_file)
        print('    lights=[',file=akr_file)
        print("    ],",file=akr_file)
        print(f'    camera=${camera},',file=akr_file)
        print(']',file=akr_file)
        print("Scene exported")


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