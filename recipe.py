import sys
import os
import useless
from useless.recipe import Recipe

if __name__ == '__main__':
    with Recipe('Release', '.') as r:
        print(sys.argv)
        print('Installing dependencies...')
        r.require('openexr')
        r.require('glm')
        r.require('cereal')
        r.require('spdlog')
        r.require('stb')
        r.require('cxxopts')
        r.require('assimp')
        r.require('pybind11')
        if 'openvdb' in sys.argv[1:]:
            r.require('openvdb')
        if 'embree' in sys.argv[1:]:
            r.require('embree')
        if 'gui' in sys.argv[1:]:
            r.require('glfw')
            r.require('glslang')
        if 'diff' in sys.argv[1:]:
            enoki = r.require('enoki')
            enoki.enable('cuda')
            enoki.enable('autodiff')
