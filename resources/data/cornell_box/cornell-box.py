mesh = AkariMesh("CornellBox-Original.obj.mesh")
materials = dict()
# OBJ Material: leftWall
mat = DiffuseMaterial()
mat.color = Color3f(0.63,0.065,0.05)
materials['leftWall'] = mat
# ======================================== 
# OBJ Material: rightWall
mat = DiffuseMaterial()
mat.color = Color3f(0.14,0.45,0.091)
materials['rightWall'] = mat
# ======================================== 
# OBJ Material: floor
mat = DiffuseMaterial()
mat.color = Color3f(0.725,0.71,0.68)
materials['floor'] = mat
# ======================================== 
# OBJ Material: ceiling
mat = DiffuseMaterial()
mat.color = Color3f(0.725,0.71,0.68)
materials['ceiling'] = mat
# ======================================== 
# OBJ Material: backWall
mat = DiffuseMaterial()
mat.color = Color3f(0.725,0.71,0.68)
materials['backWall'] = mat
# ======================================== 
# OBJ Material: shortBox
mat = DiffuseMaterial()
mat.color = Color3f(0.725,0.71,0.68)
materials['shortBox'] = mat
# ======================================== 
# OBJ Material: tallBox
mat = DiffuseMaterial()
mat.color = Color3f(0.725,0.71,0.68)
materials['tallBox'] = mat
# ======================================== 
# OBJ Material: light
mat = EmissiveMaterial()
mat.color = Color3f(17, 12, 4)
materials['light'] = mat
# ======================================== 
mesh.set_material(0, materials['leftWall'])
mesh.set_material(1, materials['rightWall'])
mesh.set_material(2, materials['floor'])
mesh.set_material(3, materials['ceiling'])
mesh.set_material(4, materials['backWall'])
mesh.set_material(5, materials['shortBox'])
mesh.set_material(6, materials['tallBox'])
mesh.set_material(7, materials['light'])
export(mesh)
