
/*

li(bounce) = li + beta* li(bounce+1);
*/
#define NRC_INPUT_ROWS ( 3 + 2 + 2 + 3 + 1 + 1)

struct NRCVertexInfo {
    vec3 x;
    vec4 ns_dir;
    vec4 albedo;
    float roughness;
    float metallic;
    vec3 li;
    vec3 beta;
};

struct NRCInferState {
    vec4 albedo;
    uint idx;
};

#define NRC_MAX_TRAIN_DEPTH 5

struct NRCTrainState {
    NRCVertexInfo vertices[5];
    uint is_training;
};

layout(set = NRC_SET, binding = 0)
buffer _NRCTrainStates {
    NRCTrainState states[];    
}NRCTrainStates;

layout(set = NRC_SET, binding = 1)
buffer _NRCTrainInputs {
    float data[];
}NRCTrainInputs;


layout(set = NRC_SET, binding = 2)
buffer _NRCTrainTargets {
    float data[];
}NRCTrainTargets;

layout(set = NRC_SET, binding = 3)
buffer _NRCInferInputs {
    float data[];
}NRCInferInputs;

layout(set = NRC_SET, binding = 4)
buffer _NRCInferOutputs {
    float data[];
}NRCInferOutputs;

layout(set = NRC_SET, binding = 5)
buffer _NRCInferState {
    NRCInferState states[];
}NRCInferStates;