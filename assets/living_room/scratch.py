import pyakari as akr
import math
scene = akr.load('base-scene.json')
print(scene.integrator)
print(scene.camera)
scene.output_path = 'out-pt.exr'
# scene.output_path = 'out-pt-owen.exr'
# scene.output_path = 'out-pt-smcmc.exr'
scene.integrator = akr.UnifiedPathTracer()
# scene.integrator = akr.SMCMC()
# scene.integrator = akr.VPL()
# scene.integrator = akr.GuidedPathTracer()
scene.integrator.spp = 4
scene.integrator.min_depth = 3
scene.integrator.max_depth = 7
scene.camera.transform.translation = akr.vec3(1.0,1.3,8.0)
scene.camera.resolution = akr.ivec2(1280//2, 720//2)
# scene.camera.resolution = akr.ivec2(1280, 720)
scene.camera.fov = math.radians(120)
floor = scene.find('Floor')
floor = [x for x in floor if isinstance(x, akr.Material)]
print(floor)
floor[0].metallic = akr.FloatTexture(0.3)
floor[0].roughness = akr.FloatTexture(0.1)
# tv = scene.find('Floor')
# tv = [x for x in tv if isinstance(x, akr.Material)]
# print(scene.find('WhitePaint'))
# white_paints = scene.find('WhitePaint')
# print(white_paints[1])
# inst = white_paints[1]
# print(inst.material.emission.value)
# inst.material.emission.value = akr.color3(0,0,0)
# akr.save(scene, 'scene-vpl.json')
akr.thread_pool_init(12)
akr.render(scene)
akr.thread_pool_finalize()