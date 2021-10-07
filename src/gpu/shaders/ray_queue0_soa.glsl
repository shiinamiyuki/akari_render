layout(set = RAY_QUEUE_SET0, binding = 0) buffer _buffer_RayQueue0_ray_o_x {
    float value[];
}buffer_RayQueue0_ray_o_x;
layout(set = RAY_QUEUE_SET0, binding = 1) buffer _buffer_RayQueue0_ray_o_y {
    float value[];
}buffer_RayQueue0_ray_o_y;
layout(set = RAY_QUEUE_SET0, binding = 2) buffer _buffer_RayQueue0_ray_o_z {
    float value[];
}buffer_RayQueue0_ray_o_z;
layout(set = RAY_QUEUE_SET0, binding = 3) buffer _buffer_RayQueue0_ray_d_x {
    float value[];
}buffer_RayQueue0_ray_d_x;
layout(set = RAY_QUEUE_SET0, binding = 4) buffer _buffer_RayQueue0_ray_d_y {
    float value[];
}buffer_RayQueue0_ray_d_y;
layout(set = RAY_QUEUE_SET0, binding = 5) buffer _buffer_RayQueue0_ray_d_z {
    float value[];
}buffer_RayQueue0_ray_d_z;
layout(set = RAY_QUEUE_SET0, binding = 6) buffer _buffer_RayQueue0_ray_tmin {
    float value[];
}buffer_RayQueue0_ray_tmin;
layout(set = RAY_QUEUE_SET0, binding = 7) buffer _buffer_RayQueue0_ray_tmax {
    float value[];
}buffer_RayQueue0_ray_tmax;
layout(set = RAY_QUEUE_SET0, binding = 8) buffer _buffer_RayQueue0_sid {
    uint value[];
}buffer_RayQueue0_sid;


void store_RayQueue0_ray_o_x(uint i, float v){
    buffer_RayQueue0_ray_o_x.value[i] = v;
} 
float load_RayQueue0_ray_o_x(uint i){
    return buffer_RayQueue0_ray_o_x.value[i];
} 

void store_RayQueue0_ray_o_y(uint i, float v){
    buffer_RayQueue0_ray_o_y.value[i] = v;
} 
float load_RayQueue0_ray_o_y(uint i){
    return buffer_RayQueue0_ray_o_y.value[i];
} 

void store_RayQueue0_ray_o_z(uint i, float v){
    buffer_RayQueue0_ray_o_z.value[i] = v;
} 
float load_RayQueue0_ray_o_z(uint i){
    return buffer_RayQueue0_ray_o_z.value[i];
} 
vec3 load_RayQueue0_ray_o(uint i) {
  vec3  ret;
  ret.x = load_RayQueue0_ray_o_x(i);
  ret.y = load_RayQueue0_ray_o_y(i);
  ret.z = load_RayQueue0_ray_o_z(i);
  return ret;
}
void store_RayQueue0_ray_o(uint i,vec3 val) {
  store_RayQueue0_ray_o_x(i, val.x);
  store_RayQueue0_ray_o_y(i, val.y);
  store_RayQueue0_ray_o_z(i, val.z);
}

void store_RayQueue0_ray_d_x(uint i, float v){
    buffer_RayQueue0_ray_d_x.value[i] = v;
} 
float load_RayQueue0_ray_d_x(uint i){
    return buffer_RayQueue0_ray_d_x.value[i];
} 

void store_RayQueue0_ray_d_y(uint i, float v){
    buffer_RayQueue0_ray_d_y.value[i] = v;
} 
float load_RayQueue0_ray_d_y(uint i){
    return buffer_RayQueue0_ray_d_y.value[i];
} 

void store_RayQueue0_ray_d_z(uint i, float v){
    buffer_RayQueue0_ray_d_z.value[i] = v;
} 
float load_RayQueue0_ray_d_z(uint i){
    return buffer_RayQueue0_ray_d_z.value[i];
} 
vec3 load_RayQueue0_ray_d(uint i) {
  vec3  ret;
  ret.x = load_RayQueue0_ray_d_x(i);
  ret.y = load_RayQueue0_ray_d_y(i);
  ret.z = load_RayQueue0_ray_d_z(i);
  return ret;
}
void store_RayQueue0_ray_d(uint i,vec3 val) {
  store_RayQueue0_ray_d_x(i, val.x);
  store_RayQueue0_ray_d_y(i, val.y);
  store_RayQueue0_ray_d_z(i, val.z);
}

void store_RayQueue0_ray_tmin(uint i, float v){
    buffer_RayQueue0_ray_tmin.value[i] = v;
} 
float load_RayQueue0_ray_tmin(uint i){
    return buffer_RayQueue0_ray_tmin.value[i];
} 

void store_RayQueue0_ray_tmax(uint i, float v){
    buffer_RayQueue0_ray_tmax.value[i] = v;
} 
float load_RayQueue0_ray_tmax(uint i){
    return buffer_RayQueue0_ray_tmax.value[i];
} 
Ray load_RayQueue0_ray(uint i) {
  Ray  ret;
  ret.o = load_RayQueue0_ray_o(i);
  ret.d = load_RayQueue0_ray_d(i);
  ret.tmin = load_RayQueue0_ray_tmin(i);
  ret.tmax = load_RayQueue0_ray_tmax(i);
  return ret;
}
void store_RayQueue0_ray(uint i,Ray val) {
  store_RayQueue0_ray_o(i, val.o);
  store_RayQueue0_ray_d(i, val.d);
  store_RayQueue0_ray_tmin(i, val.tmin);
  store_RayQueue0_ray_tmax(i, val.tmax);
}

void store_RayQueue0_sid(uint i, uint v){
    buffer_RayQueue0_sid.value[i] = v;
} 
uint load_RayQueue0_sid(uint i){
    return buffer_RayQueue0_sid.value[i];
} 
RayQueueItem load_RayQueue0(uint i) {
  RayQueueItem  ret;
  ret.ray = load_RayQueue0_ray(i);
  ret.sid = load_RayQueue0_sid(i);
  return ret;
}
void store_RayQueue0(uint i,RayQueueItem val) {
  store_RayQueue0_ray(i, val.ray);
  store_RayQueue0_sid(i, val.sid);
}
