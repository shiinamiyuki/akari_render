#ifndef QUEUE_GLSL
#define QUEUE_GLSL
#define RAY_QUEUE0 0
#define SHADOW_RAY_QUEUE 2
#define NRC_TRAIN_QUEUE 14
#define NRC_INFER_QUEUE 15
#define TOTAL_QUEUES 16

layout(set = QUEUE_COUNTER_SET,binding = 0)
buffer QueueCounters {
    uint queue_counters[];
};
#endif
