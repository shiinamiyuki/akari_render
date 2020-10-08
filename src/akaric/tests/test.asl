struct Light {
    vec3 pos;
    vec3 color;
}
const int const_ival = 2;
buffer Light[] lights;
vec3 L(){
    vec3 res = vec3(0.0);
    for(uint i = uint(0); i < lights.length; i += 1){
        res += lights[i].color;
    }
    return res;
}
(int, float) test(){
    return (1, 1.0);
}
void test3(){
    let l = L();
}
void test2(){
    let (x, y) = test();
}
