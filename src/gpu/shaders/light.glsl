#ifndef LIGHT_GLSL
#define LIGHT_GLSL
#define LIGHT_TYPE_POINT 0
#define LIGHT_TYPE_MESH 1
struct Light {
    int type;
    int index;
};
struct PointLight {
    vec3 pos;
    Texture emission;
};
struct AreaLight {
    int instance_id;
    int area_dist_id;
    Texture emission;
};
layout(set = LIGHT_SET, binding=0)
buffer _Lights {
    Light lights[];
}Lights;

layout(set = LIGHT_SET, binding=1)
buffer _PointLights {
    PointLight lights[];
}PointLights;

layout(set = LIGHT_SET, binding=2)
buffer _AreaLights {
    AreaLight lights[];
}AreaLights;

struct AliasTableEntry {
    int j;
    float t;
};

layout(set = MESH_AREA_DIST_TABLE_SET, binding=0)
buffer _MeshAreaDistributionAliasTable {
    AliasTableEntry table[];
}MeshAreaDistributionAliasTable[];
layout(set = MESH_AREA_DIST_PDF_SET, binding=0)
buffer _MeshAreaDistributionPdf {
    float pdf[];
}MeshAreaDistributionPdf[];

layout(set = LIGHT_DIST_SET, binding=0)
buffer _LightDistributionAliasTable {
    AliasTableEntry table[];
}LightDistributionAliasTable;
layout(set = LIGHT_DIST_SET, binding=1)
buffer _LightDistributionPdf {
    float pdf[];
}LightDistributionPdf;
#endif
