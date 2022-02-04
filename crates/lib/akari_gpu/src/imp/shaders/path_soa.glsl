layout(set = PATH_STATES_SET, binding = 0) buffer _buffer_PathStates_state {
    int value[];
}buffer_PathStates_state;
layout(set = PATH_STATES_SET, binding = 1) buffer _buffer_PathStates_bounce {
    int value[];
}buffer_PathStates_bounce;
layout(set = PATH_STATES_SET, binding = 2) buffer _buffer_PathStates_beta_x {
    float value[];
}buffer_PathStates_beta_x;
layout(set = PATH_STATES_SET, binding = 3) buffer _buffer_PathStates_beta_y {
    float value[];
}buffer_PathStates_beta_y;
layout(set = PATH_STATES_SET, binding = 4) buffer _buffer_PathStates_beta_z {
    float value[];
}buffer_PathStates_beta_z;
layout(set = PATH_STATES_SET, binding = 5) buffer _buffer_PathStates_l_x {
    float value[];
}buffer_PathStates_l_x;
layout(set = PATH_STATES_SET, binding = 6) buffer _buffer_PathStates_l_y {
    float value[];
}buffer_PathStates_l_y;
layout(set = PATH_STATES_SET, binding = 7) buffer _buffer_PathStates_l_z {
    float value[];
}buffer_PathStates_l_z;
layout(set = PATH_STATES_SET, binding = 8) buffer _buffer_PathStates_pixel {
    uint value[];
}buffer_PathStates_pixel;


void store_PathStates_state(uint i, int v){
    buffer_PathStates_state.value[i] = v;
} 
int load_PathStates_state(uint i){
    return buffer_PathStates_state.value[i];
} 

void store_PathStates_bounce(uint i, int v){
    buffer_PathStates_bounce.value[i] = v;
} 
int load_PathStates_bounce(uint i){
    return buffer_PathStates_bounce.value[i];
} 

void store_PathStates_beta_x(uint i, float v){
    buffer_PathStates_beta_x.value[i] = v;
} 
float load_PathStates_beta_x(uint i){
    return buffer_PathStates_beta_x.value[i];
} 

void store_PathStates_beta_y(uint i, float v){
    buffer_PathStates_beta_y.value[i] = v;
} 
float load_PathStates_beta_y(uint i){
    return buffer_PathStates_beta_y.value[i];
} 

void store_PathStates_beta_z(uint i, float v){
    buffer_PathStates_beta_z.value[i] = v;
} 
float load_PathStates_beta_z(uint i){
    return buffer_PathStates_beta_z.value[i];
} 
vec3 load_PathStates_beta(uint i) {
  vec3  ret;
  ret.x = load_PathStates_beta_x(i);
  ret.y = load_PathStates_beta_y(i);
  ret.z = load_PathStates_beta_z(i);
  return ret;
}
void store_PathStates_beta(uint i,vec3 val) {
  store_PathStates_beta_x(i, val.x);
  store_PathStates_beta_y(i, val.y);
  store_PathStates_beta_z(i, val.z);
}

void store_PathStates_l_x(uint i, float v){
    buffer_PathStates_l_x.value[i] = v;
} 
float load_PathStates_l_x(uint i){
    return buffer_PathStates_l_x.value[i];
} 

void store_PathStates_l_y(uint i, float v){
    buffer_PathStates_l_y.value[i] = v;
} 
float load_PathStates_l_y(uint i){
    return buffer_PathStates_l_y.value[i];
} 

void store_PathStates_l_z(uint i, float v){
    buffer_PathStates_l_z.value[i] = v;
} 
float load_PathStates_l_z(uint i){
    return buffer_PathStates_l_z.value[i];
} 
vec3 load_PathStates_l(uint i) {
  vec3  ret;
  ret.x = load_PathStates_l_x(i);
  ret.y = load_PathStates_l_y(i);
  ret.z = load_PathStates_l_z(i);
  return ret;
}
void store_PathStates_l(uint i,vec3 val) {
  store_PathStates_l_x(i, val.x);
  store_PathStates_l_y(i, val.y);
  store_PathStates_l_z(i, val.z);
}

void store_PathStates_pixel(uint i, uint v){
    buffer_PathStates_pixel.value[i] = v;
} 
uint load_PathStates_pixel(uint i){
    return buffer_PathStates_pixel.value[i];
} 
PathState load_PathStates(uint i) {
  PathState  ret;
  ret.state = load_PathStates_state(i);
  ret.bounce = load_PathStates_bounce(i);
  ret.beta = load_PathStates_beta(i);
  ret.l = load_PathStates_l(i);
  ret.pixel = load_PathStates_pixel(i);
  return ret;
}
void store_PathStates(uint i,PathState val) {
  store_PathStates_state(i, val.state);
  store_PathStates_bounce(i, val.bounce);
  store_PathStates_beta(i, val.beta);
  store_PathStates_l(i, val.l);
  store_PathStates_pixel(i, val.pixel);
}
