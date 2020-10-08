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


struct DiffuseBSDF{
    Spectrum R;
}
const int DiffuseBSDFType = 0;
struct GlossyBSDF{
    Spectrum R;
}
const int GlossyBSDFType = 1;
const int NumBSDF = 2;
struct BSDF {
    float [NumBSDF] weights;
    DiffuseBSDF diffuse;
    GlossyBSDF glossy;
}
