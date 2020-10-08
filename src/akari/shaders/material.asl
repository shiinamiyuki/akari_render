struct DiffuseMaterial {
    Spectrum color;
}

struct GlossyMaterial {
    Spectrum color;
}

buffer DiffuseMaterial[] diffuse_materials;
buffer GlossyMaterial[] glossy_materials;