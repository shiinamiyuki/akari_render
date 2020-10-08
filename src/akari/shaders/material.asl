struct MaterialHandle {
    int type;
    int index;
}

struct MixMaterial {
    MaterialHandle first;
    MaterialHandle second;
}
const int MixMaterialType = 0;

struct DiffuseMaterial {
    Spectrum color;
}

const int DiffuseMaterialType = 1;

struct GlossyMaterial {
    Spectrum color;
}
const int GlossyMaterialType = 2;


buffer MixMaterial[] mix_materials;
buffer DiffuseMaterial[] diffuse_materials;
buffer GlossyMaterial[] glossy_materials;


void foo(int material_type){
    switch(material_type){
        case MixMaterialType:{

        }
        default:{

        }
    }
}