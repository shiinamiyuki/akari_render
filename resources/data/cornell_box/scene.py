from akari import *
from akari.rgb import *
print(enabled_variants())
scene = Scene()
scene.commit()
print(dir(scene))