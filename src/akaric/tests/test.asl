// module Test(Spectrum, const vec3 Constant = vec3(0));
// import Another(Spectrum) as Another;

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
    return p;//hello
}
float CosTheta(vec3 w){
    return w.y;
}
float Cos2Theta(vec3 w){
    return w.y * w.y;
}
float SinTheta(vec3 w){
    return sqrt(Sin2Theta(w));
}
float max(float a,float b);
float Sin2Theta(vec3 w){
    return max(0.0, 1.0 - Cos2Theta(w));
}
float Tan2Theta(vec3 w){
    return Sin2Theta(w) / Cos2Theta(w);
}

float TanTheta(vec3 w){
    return sqrt(Tan2Theta(w));
}
float sqrt(float v);

void setZero(inout float v){
    v = 0.0;
}

float GGX_D(float alpha, vec3 m) {
    if (m.y <= 0.0)
        return 0.0;
    float a2 = alpha * alpha;
    float c2 = Cos2Theta(m);
    float t2 = Tan2Theta(m);
    float at = (a2 + t2);
    return a2 / (3.1415926 * c2 * c2 * at * at);
}
