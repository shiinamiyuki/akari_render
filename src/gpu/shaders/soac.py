from string import Template
import sys


class Output:
    def __init__(self, layouts='', sl='') -> None:
        self.sl = str(sl)
        self.layouts = str(layouts)

    def append(self, rhs):
        self.sl += rhs.sl
        self.layouts += rhs.layouts

    def __iadd__(self, rhs):
        self.append(rhs)
        return self


soa_template1 = Template(
    '''layout(set = $set, binding = $binding) buffer _$buffer_name {
    $ty value[];
}$buffer_name;
''')
soa_template2 = Template('''
void store_$name(uint i, $ty v){
    $buffer_name.value[i] = v;
} 
$ty load_$name(uint i){
    return $buffer_name.value[i];
} 
''')


class Binding:
    def __init__(self, binding) -> None:
        self.binding = binding

    def next(self):
        binding = self.binding
        self.binding += 1
        return binding


def def_soa(name: str, ty: str, set_, binding: Binding):

    d = {
        'name': name,
        'ty': ty,
        'set': set_,
        'binding': binding.next(),
        'buffer_name': 'buffer_' + name
    }
    return Output(soa_template1.substitute(d), soa_template2.substitute(d))


def def_soa_vec(name: str, ty: str, count: int, set_, binding: Binding):
    components = 'xyzw'
    out = Output()
    for i in range(count):
        out += def_soa(name + '_' + components[i], 'float', set_, binding)
    vec_ty = ''
    if ty == 'float':
        vec_ty = 'vec' + str(count)
    else:
        assert False
    out.sl += def_sl(name, vec_ty, components[:count])
    return out


def def_soa_vec3(name: str, set_, binding: Binding):
    return def_soa_vec(name, 'float', 3, set_, binding)


def def_sl(name: str, ty: str, fields: list):
    out = ''
    out += ty + ' load_' + name + '(uint i) {\n'
    out += '  ' + ty + ' ' + ' ret;\n'
    for field in fields:
        out += '  ret.' + field + ' = load_' + name + '_' + field + '(i);\n'
    out += '  return ret;\n}\n'

    out += 'void store_' + name + '(uint i,' + ty + ' val) {\n'
    for field in fields:
        out += '  store_' + name + '_' + field + '(i, val.' + field + ');\n'
    out += '}\n'
    return out


def def_soa_ray(name: str, set_, binding):
    out = ''
    out += def_soa_vec3(name + '_o', set_, binding)
    out += def_soa_vec3(name + '_d', set_, binding)
    out += def_soa(name + '_tmin', 'float', set_, binding)
    out += def_soa(name + '_tmax', 'float', set_, binding)
#     out += '''Ray load_##name##(int i) {
#     Ray ray;
#     ray.o = load_##name##_o(i);
#     ray.d = load_##name##_d(i);
#     ray.tmin = load_##name##_tmin(i);
#     ray.tmax = load_##name##_tmax(i);
#     return ray;
# }'''.replace('##name##', name)
    out += def_sl(name, 'Ray', ['o', 'd', 'tmin', 'tmax'])
    return out

# SHADOW_QUEUE_SET = 1

# soa_gen = ''
# soa_gen += def_soa_ray('shadow_ray', SHADOW_QUEUE_SET, Binding(0))


def gen(cfg):
    # src_file = sys.argv[1]
    # with open(src_file, 'r') as f:
    #     src = f.read()
    # cfg = eval(src)
    flats = ['float', 'int', 'uint']
    vecs = ['vec2', 'vec3', 'vec4']
    vars = cfg['vars']
    types = cfg['types']
    out = Output()
    bindings = dict()

    def generate_binding(name: str, ty: str, set: int):
        if ty in flats:
            return def_soa(name, ty, set, bindings[set])
        elif ty in vecs:
            n = int(ty[-1])
            return def_soa_vec(name, 'float', n, set, bindings[set])
        else:
            struct_type = types[ty]
            out = Output()
            for fieldname, fieldtype in struct_type.items():
                out += generate_binding(name + '_' + fieldname, fieldtype, set)
            out.sl += def_sl(name, ty, [x for x in struct_type])
            return out
    for varname, var in vars.items():
        if var['set'] not in bindings:
            bindings[var['set']] = Binding(0)
        ty = var['type']
        out += generate_binding(varname, ty, var['set'])

    return (out.layouts + '\n' + out.sl)
    # with open(sys.argv[2], 'w') as f:
    # f.write(out.layouts + '\n' + out.sl)


types = {
    "Ray": {
        "o": "vec3",
        "d": "vec3",
        "tmin": "float",
        "tmax": "float"
    },
    "ShadowQueueItem": {
        "ray": "Ray",
        "ld":"vec3",
        "sid": "uint"
    },
    "RayQueueItem": {
        "ray": "Ray",
        "sid": "uint"
    },
    "PathState": {
        "state": "int",
        "bounce": "int",
        "beta": "vec3",
        "l": "vec3",
        "pixel":"uint",
    },
    "MaterialEvalInfo":{
        "wo":"vec3",
        "p":"vec3",
        "ng":"vec3",
        "ns":"vec3",
        "texcoords":"vec2",
        "bsdf":"int",
    }
}

material_eval_info_soa = {
    "MaterialEvalInfos":{
        "set":"MATERIAL_EVAL_SET",
        "type":"MaterialEvalInfo",
    }
}
shadow_queue = {
    "ShadowQueue":{
        "set":"SHADOW_QUEUE_SET",
        "type":"ShadowQueueItem"
    }
}

pt_soa = {
    "PathStates": {
        "type": "PathState",
        "set": 'PATH_STATES_SET'
    },
    # "RayQueueCur":{
    #     "type":"RayQueueItem",
    #     "set":5
    # },
    # "RayQueueNext":{
    #     "type":"RayQueueItem",
    #     "set":6
    # }
}

with open('ray_queue0_soa.glsl', 'w') as f:
    f.write(gen({
        "types": types,
        "vars": {"RayQueue0": {
            "type": "RayQueueItem",
            "set": 'RAY_QUEUE_SET0'
        }}
    }))
with open('ray_queue1_soa.glsl', 'w') as f:
    f.write(gen({
        "types": types,
        "vars": {"RayQueue1": {
            "type": "RayQueueItem",
            "set": 'RAY_QUEUE_SET1'
        }}
    }))
with open('path_soa.glsl', 'w') as f:
    f.write(gen({
        "types":types,
        "vars":pt_soa
    }))
with open('material_eval_info.glsl', 'w') as f:
    f.write(gen({
        "types":types,
        "vars":material_eval_info_soa
    }))
with open('shadow_queue.glsl', 'w') as f:
    f.write(gen({
        "types":types,
        "vars":shadow_queue
    }))