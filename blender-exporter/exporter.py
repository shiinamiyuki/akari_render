import bpy
import bmesh
import json
import bpy.types as T
import sys
from typing import List, Tuple, Dict
from enum import Enum, auto
import os
import struct

D = bpy.data
C = bpy.context
depsgraph = C.evaluated_depsgraph_get()
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
argv = sys.argv
print(argv)
argv_offset = argv.index("--") + 1
argv = argv[argv_offset:]
force_export ='--force' in argv or '-f' in argv
update_mesh_only = '--update-mesh-only' in argv or '-u' in argv
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
MESH_DIR = os.path.join(OUT_DIR, "meshes")
if not os.path.exists(MESH_DIR):
    os.makedirs(MESH_DIR)

def convert_coord_sys(v):
    # blender is z-up, right-handed
    # we are y-up, right-handed
    return (v[0], v[2], -v[1])

class UniqueName:
    def __init__(self):
        self.m = dict()
        self.names = dict()

    def contains(self, m):
        return m in self.m

    def get(self, m):
        if m in self.m:
            return self.m
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
visited = UniqueName()

class SceneExporter:
    def __init__(self, scene):
        self.scene = scene
        self.exported_meshes = list()

    def visible_objects(self):
        return [ob for ob in self.scene.objects if not ob.hide_render]

    def export_node_tree(self, node_tree) -> List[dict]:
        # exporter = NodeTreeExporter()
        # return exporter.export(node_tree)
        pass


    def export_material(self, m):
        if m is None:
            return
        if visited.contains(m):
            return
        name = visited.get(m)
        print(f"Exporting Material `{m.name}` -> {name}")
        print(m.node_tree)
        if m.node_tree is not None:
           self.export_node_tree(m.node_tree)

    def export_mesh(self, obj):
        if visited.contains(obj):
            return
        name = visited.get(obj)
        print(f"Exporting Mesh `{obj.name}` -> {name}")
        print(obj.data)
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
        for v in V:
            vert_buffer.write(struct.pack("fff", *convert_coord_sys(v.co)))
        for f in F:
            indices = f.vertices
            ind_buffer.write(struct.pack("III", indices[0], indices[1], indices[2]))
            normals = f.split_normals
            for n in normals:
                normal_buffer.write(struct.pack("fff", *convert_coord_sys(n)))
            for loop_index in f.loops:
                if has_uv:
                    uv = m.uv_layers[0].uv[loop_index].vector
                    uv_buffer.write(struct.pack("ff", uv[0], uv[1]))
                    tangent = m.loops[loop_index].tangent
                    tangent_buffer.write(struct.pack("fff", *convert_coord_sys(tangent)))
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
        print(f'${name}_vert = Buffer[name="vert", path="{os.path.join(MESH_DIR, f"{name}.vert")}"]',file=akr_file)
        print(f'${name}_ind = Buffer[name="ind", path="{os.path.join(MESH_DIR, f"{name}.ind")}"]',file=akr_file)
        print(f'${name}_normal = Buffer[name="normal", path="{os.path.join(MESH_DIR, f"{name}.normal")}]',file=akr_file)
        if has_uv:
            print(f'${name}_uv = Buffer[name="uv", path="{os.path.join(MESH_DIR, f"{name}.uv")}"]',file=akr_file)
            print(f'${name}_tangent = Buffer[name="tangent", path="{os.path.join(MESH_DIR, f"{name}.tangent")}"]',file=akr_file)
            print(f'${name}_bitangent = Buffer[name="bitangent", path="{os.path.join(MESH_DIR, f"{name}.bitangent")}"]',file=akr_file)
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
        print("    ]\n]",file=akr_file)
        for mat in obj.data.materials:
            self.export_material(mat)
        self.exported_meshes.append(name)

    def export(self):
        name = visited.get(self.scene)
        # seperate meshes by material
        for obj in self.visible_objects():
            if obj.type == 'MESH':
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
        print(f'${name} = Scene[geometries=[',file=akr_file)
        for i, m in enumerate(self.exported_meshes):
            print(f'    ${m}',file=akr_file, end="")
            if i != len(self.exported_meshes) - 1:
                print(",",file=akr_file)
            else:
                print(file=akr_file)
        print("]]",file=akr_file)
        print("Scene exported")


def export_scene(scene):
    exporter = SceneExporter(scene)
    exporter.export()
export_scene(C.scene)
akr_file.close()

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "modified.blend"))