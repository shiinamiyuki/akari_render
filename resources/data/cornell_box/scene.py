from akari import *
from akari.rgb import *
print(enabled_variants())
scene = Scene()
cbox = OBJMesh("CornellBox-Original.obj")
cbox.commit()
scene.camera = PerspectiveCamera()
scene.shapes.append(cbox)
scene.commit()
print(dir(scene))