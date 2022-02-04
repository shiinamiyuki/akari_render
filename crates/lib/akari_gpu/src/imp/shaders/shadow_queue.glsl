layout(set = SHADOW_QUEUE_SET, binding = 0) buffer _buffer_ShadowQueue_ray_o_x {
    float value[];
}buffer_ShadowQueue_ray_o_x;
layout(set = SHADOW_QUEUE_SET, binding = 1) buffer _buffer_ShadowQueue_ray_o_y {
    float value[];
}buffer_ShadowQueue_ray_o_y;
layout(set = SHADOW_QUEUE_SET, binding = 2) buffer _buffer_ShadowQueue_ray_o_z {
    float value[];
}buffer_ShadowQueue_ray_o_z;
layout(set = SHADOW_QUEUE_SET, binding = 3) buffer _buffer_ShadowQueue_ray_d_x {
    float value[];
}buffer_ShadowQueue_ray_d_x;
layout(set = SHADOW_QUEUE_SET, binding = 4) buffer _buffer_ShadowQueue_ray_d_y {
    float value[];
}buffer_ShadowQueue_ray_d_y;
layout(set = SHADOW_QUEUE_SET, binding = 5) buffer _buffer_ShadowQueue_ray_d_z {
    float value[];
}buffer_ShadowQueue_ray_d_z;
layout(set = SHADOW_QUEUE_SET, binding = 6) buffer _buffer_ShadowQueue_ray_tmin {
    float value[];
}buffer_ShadowQueue_ray_tmin;
layout(set = SHADOW_QUEUE_SET, binding = 7) buffer _buffer_ShadowQueue_ray_tmax {
    float value[];
}buffer_ShadowQueue_ray_tmax;
layout(set = SHADOW_QUEUE_SET, binding = 8) buffer _buffer_ShadowQueue_ld_x {
    float value[];
}buffer_ShadowQueue_ld_x;
layout(set = SHADOW_QUEUE_SET, binding = 9) buffer _buffer_ShadowQueue_ld_y {
    float value[];
}buffer_ShadowQueue_ld_y;
layout(set = SHADOW_QUEUE_SET, binding = 10) buffer _buffer_ShadowQueue_ld_z {
    float value[];
}buffer_ShadowQueue_ld_z;
layout(set = SHADOW_QUEUE_SET, binding = 11) buffer _buffer_ShadowQueue_sid {
    uint value[];
}buffer_ShadowQueue_sid;


void store_ShadowQueue_ray_o_x(uint i, float v){
    buffer_ShadowQueue_ray_o_x.value[i] = v;
} 
float load_ShadowQueue_ray_o_x(uint i){
    return buffer_ShadowQueue_ray_o_x.value[i];
} 

void store_ShadowQueue_ray_o_y(uint i, float v){
    buffer_ShadowQueue_ray_o_y.value[i] = v;
} 
float load_ShadowQueue_ray_o_y(uint i){
    return buffer_ShadowQueue_ray_o_y.value[i];
} 

void store_ShadowQueue_ray_o_z(uint i, float v){
    buffer_ShadowQueue_ray_o_z.value[i] = v;
} 
float load_ShadowQueue_ray_o_z(uint i){
    return buffer_ShadowQueue_ray_o_z.value[i];
} 
vec3 load_ShadowQueue_ray_o(uint i) {
  vec3  ret;
  ret.x = load_ShadowQueue_ray_o_x(i);
  ret.y = load_ShadowQueue_ray_o_y(i);
  ret.z = load_ShadowQueue_ray_o_z(i);
  return ret;
}
void store_ShadowQueue_ray_o(uint i,vec3 val) {
  store_ShadowQueue_ray_o_x(i, val.x);
  store_ShadowQueue_ray_o_y(i, val.y);
  store_ShadowQueue_ray_o_z(i, val.z);
}

void store_ShadowQueue_ray_d_x(uint i, float v){
    buffer_ShadowQueue_ray_d_x.value[i] = v;
} 
float load_ShadowQueue_ray_d_x(uint i){
    return buffer_ShadowQueue_ray_d_x.value[i];
} 

void store_ShadowQueue_ray_d_y(uint i, float v){
    buffer_ShadowQueue_ray_d_y.value[i] = v;
} 
float load_ShadowQueue_ray_d_y(uint i){
    return buffer_ShadowQueue_ray_d_y.value[i];
} 

void store_ShadowQueue_ray_d_z(uint i, float v){
    buffer_ShadowQueue_ray_d_z.value[i] = v;
} 
float load_ShadowQueue_ray_d_z(uint i){
    return buffer_ShadowQueue_ray_d_z.value[i];
} 
vec3 load_ShadowQueue_ray_d(uint i) {
  vec3  ret;
  ret.x = load_ShadowQueue_ray_d_x(i);
  ret.y = load_ShadowQueue_ray_d_y(i);
  ret.z = load_ShadowQueue_ray_d_z(i);
  return ret;
}
void store_ShadowQueue_ray_d(uint i,vec3 val) {
  store_ShadowQueue_ray_d_x(i, val.x);
  store_ShadowQueue_ray_d_y(i, val.y);
  store_ShadowQueue_ray_d_z(i, val.z);
}

void store_ShadowQueue_ray_tmin(uint i, float v){
    buffer_ShadowQueue_ray_tmin.value[i] = v;
} 
float load_ShadowQueue_ray_tmin(uint i){
    return buffer_ShadowQueue_ray_tmin.value[i];
} 

void store_ShadowQueue_ray_tmax(uint i, float v){
    buffer_ShadowQueue_ray_tmax.value[i] = v;
} 
float load_ShadowQueue_ray_tmax(uint i){
    return buffer_ShadowQueue_ray_tmax.value[i];
} 
Ray load_ShadowQueue_ray(uint i) {
  Ray  ret;
  ret.o = load_ShadowQueue_ray_o(i);
  ret.d = load_ShadowQueue_ray_d(i);
  ret.tmin = load_ShadowQueue_ray_tmin(i);
  ret.tmax = load_ShadowQueue_ray_tmax(i);
  return ret;
}
void store_ShadowQueue_ray(uint i,Ray val) {
  store_ShadowQueue_ray_o(i, val.o);
  store_ShadowQueue_ray_d(i, val.d);
  store_ShadowQueue_ray_tmin(i, val.tmin);
  store_ShadowQueue_ray_tmax(i, val.tmax);
}

void store_ShadowQueue_ld_x(uint i, float v){
    buffer_ShadowQueue_ld_x.value[i] = v;
} 
float load_ShadowQueue_ld_x(uint i){
    return buffer_ShadowQueue_ld_x.value[i];
} 

void store_ShadowQueue_ld_y(uint i, float v){
    buffer_ShadowQueue_ld_y.value[i] = v;
} 
float load_ShadowQueue_ld_y(uint i){
    return buffer_ShadowQueue_ld_y.value[i];
} 

void store_ShadowQueue_ld_z(uint i, float v){
    buffer_ShadowQueue_ld_z.value[i] = v;
} 
float load_ShadowQueue_ld_z(uint i){
    return buffer_ShadowQueue_ld_z.value[i];
} 
vec3 load_ShadowQueue_ld(uint i) {
  vec3  ret;
  ret.x = load_ShadowQueue_ld_x(i);
  ret.y = load_ShadowQueue_ld_y(i);
  ret.z = load_ShadowQueue_ld_z(i);
  return ret;
}
void store_ShadowQueue_ld(uint i,vec3 val) {
  store_ShadowQueue_ld_x(i, val.x);
  store_ShadowQueue_ld_y(i, val.y);
  store_ShadowQueue_ld_z(i, val.z);
}

void store_ShadowQueue_sid(uint i, uint v){
    buffer_ShadowQueue_sid.value[i] = v;
} 
uint load_ShadowQueue_sid(uint i){
    return buffer_ShadowQueue_sid.value[i];
} 
ShadowQueueItem load_ShadowQueue(uint i) {
  ShadowQueueItem  ret;
  ret.ray = load_ShadowQueue_ray(i);
  ret.ld = load_ShadowQueue_ld(i);
  ret.sid = load_ShadowQueue_sid(i);
  return ret;
}
void store_ShadowQueue(uint i,ShadowQueueItem val) {
  store_ShadowQueue_ray(i, val.ray);
  store_ShadowQueue_ld(i, val.ld);
  store_ShadowQueue_sid(i, val.sid);
}
