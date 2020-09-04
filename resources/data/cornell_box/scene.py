from akari import *
from akari.rgb import *
print(enabled_variants())
scene = Scene()
scene.output = "out.png"
cbox = load_fragment("cornell-box.py")
# cbox = OBJMesh('CornellBox-Original.obj')
camera = PerspectiveCamera()
camera.fov = radians(15)
camera.position = Point3f(0, 1, 9)
scene.camera = camera
scene.add_mesh(cbox)
scene.integrator = Path()
scene.integrator.spp = 16
scene.render()

