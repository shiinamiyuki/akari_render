import pyakari as akr
import math
scene = akr.load('scene.json')
print(scene.integrator)
print(scene.camera)
# scene.integrator = akr.PathTracer()
# scene.integrator = akr.SMCMC()
# scene.integrator = akr.MCMC()
# scene.integrator = akr.VPL()
# scene.output_path = 'out-pt.exr'
scene.output_path = 'out-ppg.exr'
# scene.output_path = 'out-smcmc.exr'
scene.integrator = akr.GuidedPathTracer()
# scene.integrator.metropolized = True
scene.integrator.spp = 1024
scene.integrator.min_depth = 3
scene.integrator.max_depth = 7
scene.camera.transform.translation = akr.vec3(0, 0.8, 9.0)
scene.camera.resolution = akr.ivec2(512, 400)
# scene.camera.resolution = akr.ivec2(512*2, 400*2)
scene.camera.fov = math.radians(15)

akr.thread_pool_init(12)
akr.render(scene)
akr.thread_pool_finalize()