
vec3 foo(){
    return vec3(0.0);
}
vec3 pow4(vec3 x){
    int i =0;
    vec3 p = vec3(1.0);
    while(i < 4){
        p = p * x;
        i = i + 1;
    }
    return p;
}