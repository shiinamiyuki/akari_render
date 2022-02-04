layout(set = MATERIAL_EVAL_SET, binding = 0) buffer _buffer_MaterialEvalInfos_wo_x {
    float value[];
}buffer_MaterialEvalInfos_wo_x;
layout(set = MATERIAL_EVAL_SET, binding = 1) buffer _buffer_MaterialEvalInfos_wo_y {
    float value[];
}buffer_MaterialEvalInfos_wo_y;
layout(set = MATERIAL_EVAL_SET, binding = 2) buffer _buffer_MaterialEvalInfos_wo_z {
    float value[];
}buffer_MaterialEvalInfos_wo_z;
layout(set = MATERIAL_EVAL_SET, binding = 3) buffer _buffer_MaterialEvalInfos_p_x {
    float value[];
}buffer_MaterialEvalInfos_p_x;
layout(set = MATERIAL_EVAL_SET, binding = 4) buffer _buffer_MaterialEvalInfos_p_y {
    float value[];
}buffer_MaterialEvalInfos_p_y;
layout(set = MATERIAL_EVAL_SET, binding = 5) buffer _buffer_MaterialEvalInfos_p_z {
    float value[];
}buffer_MaterialEvalInfos_p_z;
layout(set = MATERIAL_EVAL_SET, binding = 6) buffer _buffer_MaterialEvalInfos_ng_x {
    float value[];
}buffer_MaterialEvalInfos_ng_x;
layout(set = MATERIAL_EVAL_SET, binding = 7) buffer _buffer_MaterialEvalInfos_ng_y {
    float value[];
}buffer_MaterialEvalInfos_ng_y;
layout(set = MATERIAL_EVAL_SET, binding = 8) buffer _buffer_MaterialEvalInfos_ng_z {
    float value[];
}buffer_MaterialEvalInfos_ng_z;
layout(set = MATERIAL_EVAL_SET, binding = 9) buffer _buffer_MaterialEvalInfos_ns_x {
    float value[];
}buffer_MaterialEvalInfos_ns_x;
layout(set = MATERIAL_EVAL_SET, binding = 10) buffer _buffer_MaterialEvalInfos_ns_y {
    float value[];
}buffer_MaterialEvalInfos_ns_y;
layout(set = MATERIAL_EVAL_SET, binding = 11) buffer _buffer_MaterialEvalInfos_ns_z {
    float value[];
}buffer_MaterialEvalInfos_ns_z;
layout(set = MATERIAL_EVAL_SET, binding = 12) buffer _buffer_MaterialEvalInfos_texcoords_x {
    float value[];
}buffer_MaterialEvalInfos_texcoords_x;
layout(set = MATERIAL_EVAL_SET, binding = 13) buffer _buffer_MaterialEvalInfos_texcoords_y {
    float value[];
}buffer_MaterialEvalInfos_texcoords_y;
layout(set = MATERIAL_EVAL_SET, binding = 14) buffer _buffer_MaterialEvalInfos_bsdf {
    int value[];
}buffer_MaterialEvalInfos_bsdf;


void store_MaterialEvalInfos_wo_x(uint i, float v){
    buffer_MaterialEvalInfos_wo_x.value[i] = v;
} 
float load_MaterialEvalInfos_wo_x(uint i){
    return buffer_MaterialEvalInfos_wo_x.value[i];
} 

void store_MaterialEvalInfos_wo_y(uint i, float v){
    buffer_MaterialEvalInfos_wo_y.value[i] = v;
} 
float load_MaterialEvalInfos_wo_y(uint i){
    return buffer_MaterialEvalInfos_wo_y.value[i];
} 

void store_MaterialEvalInfos_wo_z(uint i, float v){
    buffer_MaterialEvalInfos_wo_z.value[i] = v;
} 
float load_MaterialEvalInfos_wo_z(uint i){
    return buffer_MaterialEvalInfos_wo_z.value[i];
} 
vec3 load_MaterialEvalInfos_wo(uint i) {
  vec3  ret;
  ret.x = load_MaterialEvalInfos_wo_x(i);
  ret.y = load_MaterialEvalInfos_wo_y(i);
  ret.z = load_MaterialEvalInfos_wo_z(i);
  return ret;
}
void store_MaterialEvalInfos_wo(uint i,vec3 val) {
  store_MaterialEvalInfos_wo_x(i, val.x);
  store_MaterialEvalInfos_wo_y(i, val.y);
  store_MaterialEvalInfos_wo_z(i, val.z);
}

void store_MaterialEvalInfos_p_x(uint i, float v){
    buffer_MaterialEvalInfos_p_x.value[i] = v;
} 
float load_MaterialEvalInfos_p_x(uint i){
    return buffer_MaterialEvalInfos_p_x.value[i];
} 

void store_MaterialEvalInfos_p_y(uint i, float v){
    buffer_MaterialEvalInfos_p_y.value[i] = v;
} 
float load_MaterialEvalInfos_p_y(uint i){
    return buffer_MaterialEvalInfos_p_y.value[i];
} 

void store_MaterialEvalInfos_p_z(uint i, float v){
    buffer_MaterialEvalInfos_p_z.value[i] = v;
} 
float load_MaterialEvalInfos_p_z(uint i){
    return buffer_MaterialEvalInfos_p_z.value[i];
} 
vec3 load_MaterialEvalInfos_p(uint i) {
  vec3  ret;
  ret.x = load_MaterialEvalInfos_p_x(i);
  ret.y = load_MaterialEvalInfos_p_y(i);
  ret.z = load_MaterialEvalInfos_p_z(i);
  return ret;
}
void store_MaterialEvalInfos_p(uint i,vec3 val) {
  store_MaterialEvalInfos_p_x(i, val.x);
  store_MaterialEvalInfos_p_y(i, val.y);
  store_MaterialEvalInfos_p_z(i, val.z);
}

void store_MaterialEvalInfos_ng_x(uint i, float v){
    buffer_MaterialEvalInfos_ng_x.value[i] = v;
} 
float load_MaterialEvalInfos_ng_x(uint i){
    return buffer_MaterialEvalInfos_ng_x.value[i];
} 

void store_MaterialEvalInfos_ng_y(uint i, float v){
    buffer_MaterialEvalInfos_ng_y.value[i] = v;
} 
float load_MaterialEvalInfos_ng_y(uint i){
    return buffer_MaterialEvalInfos_ng_y.value[i];
} 

void store_MaterialEvalInfos_ng_z(uint i, float v){
    buffer_MaterialEvalInfos_ng_z.value[i] = v;
} 
float load_MaterialEvalInfos_ng_z(uint i){
    return buffer_MaterialEvalInfos_ng_z.value[i];
} 
vec3 load_MaterialEvalInfos_ng(uint i) {
  vec3  ret;
  ret.x = load_MaterialEvalInfos_ng_x(i);
  ret.y = load_MaterialEvalInfos_ng_y(i);
  ret.z = load_MaterialEvalInfos_ng_z(i);
  return ret;
}
void store_MaterialEvalInfos_ng(uint i,vec3 val) {
  store_MaterialEvalInfos_ng_x(i, val.x);
  store_MaterialEvalInfos_ng_y(i, val.y);
  store_MaterialEvalInfos_ng_z(i, val.z);
}

void store_MaterialEvalInfos_ns_x(uint i, float v){
    buffer_MaterialEvalInfos_ns_x.value[i] = v;
} 
float load_MaterialEvalInfos_ns_x(uint i){
    return buffer_MaterialEvalInfos_ns_x.value[i];
} 

void store_MaterialEvalInfos_ns_y(uint i, float v){
    buffer_MaterialEvalInfos_ns_y.value[i] = v;
} 
float load_MaterialEvalInfos_ns_y(uint i){
    return buffer_MaterialEvalInfos_ns_y.value[i];
} 

void store_MaterialEvalInfos_ns_z(uint i, float v){
    buffer_MaterialEvalInfos_ns_z.value[i] = v;
} 
float load_MaterialEvalInfos_ns_z(uint i){
    return buffer_MaterialEvalInfos_ns_z.value[i];
} 
vec3 load_MaterialEvalInfos_ns(uint i) {
  vec3  ret;
  ret.x = load_MaterialEvalInfos_ns_x(i);
  ret.y = load_MaterialEvalInfos_ns_y(i);
  ret.z = load_MaterialEvalInfos_ns_z(i);
  return ret;
}
void store_MaterialEvalInfos_ns(uint i,vec3 val) {
  store_MaterialEvalInfos_ns_x(i, val.x);
  store_MaterialEvalInfos_ns_y(i, val.y);
  store_MaterialEvalInfos_ns_z(i, val.z);
}

void store_MaterialEvalInfos_texcoords_x(uint i, float v){
    buffer_MaterialEvalInfos_texcoords_x.value[i] = v;
} 
float load_MaterialEvalInfos_texcoords_x(uint i){
    return buffer_MaterialEvalInfos_texcoords_x.value[i];
} 

void store_MaterialEvalInfos_texcoords_y(uint i, float v){
    buffer_MaterialEvalInfos_texcoords_y.value[i] = v;
} 
float load_MaterialEvalInfos_texcoords_y(uint i){
    return buffer_MaterialEvalInfos_texcoords_y.value[i];
} 
vec2 load_MaterialEvalInfos_texcoords(uint i) {
  vec2  ret;
  ret.x = load_MaterialEvalInfos_texcoords_x(i);
  ret.y = load_MaterialEvalInfos_texcoords_y(i);
  return ret;
}
void store_MaterialEvalInfos_texcoords(uint i,vec2 val) {
  store_MaterialEvalInfos_texcoords_x(i, val.x);
  store_MaterialEvalInfos_texcoords_y(i, val.y);
}

void store_MaterialEvalInfos_bsdf(uint i, int v){
    buffer_MaterialEvalInfos_bsdf.value[i] = v;
} 
int load_MaterialEvalInfos_bsdf(uint i){
    return buffer_MaterialEvalInfos_bsdf.value[i];
} 
MaterialEvalInfo load_MaterialEvalInfos(uint i) {
  MaterialEvalInfo  ret;
  ret.wo = load_MaterialEvalInfos_wo(i);
  ret.p = load_MaterialEvalInfos_p(i);
  ret.ng = load_MaterialEvalInfos_ng(i);
  ret.ns = load_MaterialEvalInfos_ns(i);
  ret.texcoords = load_MaterialEvalInfos_texcoords(i);
  ret.bsdf = load_MaterialEvalInfos_bsdf(i);
  return ret;
}
void store_MaterialEvalInfos(uint i,MaterialEvalInfo val) {
  store_MaterialEvalInfos_wo(i, val.wo);
  store_MaterialEvalInfos_p(i, val.p);
  store_MaterialEvalInfos_ng(i, val.ng);
  store_MaterialEvalInfos_ns(i, val.ns);
  store_MaterialEvalInfos_texcoords(i, val.texcoords);
  store_MaterialEvalInfos_bsdf(i, val.bsdf);
}
