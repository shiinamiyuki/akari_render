# Akari Shading Language

## Example

```glsl
struct ShadingPoint {
    vec2 texcoords;
    vec2 uv;
};
// float[3] is the same as vec3
using Spectrum = float[3];

Spectrum main(in ShadingPoint sp){
    return Spectrum(length(sp.texcoords));
}

```

## Features
No function overload
No implicit cast