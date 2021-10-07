#version 460
#define QUEUE_COUNTER_SET 0
#include "queue.glsl"

void main() {
    if (gl_GlobalInvocationID.x != 0)
        return;
    for(int i =0;i<TOTAL_QUEUES;i++){
        queue_counters[i] = 0;
    }
}