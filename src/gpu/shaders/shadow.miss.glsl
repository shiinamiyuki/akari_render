#version 460
#extension GL_NV_ray_tracing : require
layout(location = 0) rayPayloadInNV bool occluded;
void main() {
    occluded = false;
}