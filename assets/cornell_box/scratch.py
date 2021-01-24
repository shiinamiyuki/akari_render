import pyakari as akr
import math
scene = akr.load('scene.json')
# print(scene.integrator)
# print(scene.camera)
# print([x for x in scene.find('shortBox') if isinstance(x, akr.Instance)])
# exit(0)
box = [x for x in scene.find('shortBox') if isinstance(x, akr.Instance)][0]
box.material = None
box.volume = akr.HomogeneousVolume()
box.volume.density = 10
# scene.output_path = 'out-ppg.exr'
# scene.output_path = 'out-mlt-ppg.png'
scene.output_path = 'out-pt.exr'
# scene.output_path = 'out-smcmc.exr'
# scene.integrator = akr.PathTracer()
scene.integrator = akr.UnifiedPathTracer()
# scene.integrator = akr.SMCMC()
# scene.integrator = akr.VPL()
# scene.integrator = akr.GuidedPathTracer()
# scene.integrator.metropolized = True
scene.integrator.spp = 4
scene.integrator.min_depth = 3
scene.integrator.max_depth = 7
scene.camera.transform.translation = akr.vec3(0,1,9.0)
scene.camera.resolution = akr.ivec2(512, 512)
scene.camera.fov = math.radians(15)
# scene.camera.resolution = akr.ivec2(1280, 720)
akr.thread_pool_init(12)
akr.render(scene)
akr.thread_pool_finalize()