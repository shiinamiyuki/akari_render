from akari import *
from akari.rgb import *
print(enabled_variants())
scene = Scene()
scene.output = "out.png"
cbox = OBJMesh("CornellBox-Original.obj")
camera = PerspectiveCamera()
camera.fov = radians(15)
camera.position = Point3f(0, 1, 9)
scene.camera = camera
scene.add_mesh(cbox)
scene.render()

