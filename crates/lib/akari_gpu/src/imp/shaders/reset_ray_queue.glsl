#version 460
#define QUEUE_COUNTER_SET 0
#include "queue.glsl"

void main() {
    if (gl_GlobalInvocationID.x != 0)
        return;
    queue_counters[SHADOW_RAY_QUEUE] = 0;
    queue_counters[RAY_QUEUE0] = 0;
}