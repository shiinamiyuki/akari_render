import sys
types_s =\
'''
 struct Ray
 struct Intersection
 struct Triangle
 struct ShadingPoint
 class  BSDF
 class  BSDFComponent
 class  Material
 class  EndPoint
 class  Camera
 class  Sampler
 class  Texture
 class  Integrator
 struct SurfaceSample
 struct Interaction
 struct SurfaceInteraction
 struct EndPointInteraction
 struct VolumeInteraction
 struct MaterialSlot
 class  Light
 class  Mesh
 struct Emission
 class  Scene
 struct BSDFSample
'''

types = [ y for y in [x.strip() for x in types_s.splitlines()] if y]
types_name = [x.split()[1] for x in types]
prolog = \
'''#pragma once
namespace akari{
'''
epilog = '''
}
'''
def gen_fwd():
    s = ''
    for ty in types:
        s += '    template<typename Float, typename Spectrum> ' + ty + ';\n'
    return s

def gen_alias():
    s = '    template<typename Float, typename Spectrum> struct Collection {\n'
    for ty in types_name:
        s += '        using ' + ty + ' = ' + ty + '<Float, Spectrum>;\n'
    s += '    };\n'
    return s

# print(prolog + gen_fwd() + gen_alias() + epilog)
if __name__ == '__main__':
    with open(sys.argv[1], 'w') as f:
        f.write(prolog + gen_fwd() + gen_alias() + epilog)