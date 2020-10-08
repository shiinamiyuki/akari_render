__builtin__ vec2 length(vec2 v);
__builtin__ vec3 length(vec3 v);
__builtin__ vec4 length(vec4 v);

__builtin__ vec2 normalize(vec2 v);
__builtin__ vec3 normalize(vec3 v);
__builtin__ vec4 normalize(vec4 v);

__builtin__ float dot(vec2 u, vec2 v);
__builtin__ float dot(vec3 u, vec3 v);
__builtin__ float dot(vec4 u, vec4 v);
__builtin__ int dot(ivec2 u, ivec2 v);
__builtin__ int dot(ivec3 u, ivec3 v);
__builtin__ int dot(ivec4 u, ivec4 v);

__builtin__ vec3 cross(vec3 u, vec3 v);

__builtin__ vec2 min(vec2 u, vec2 v);
__builtin__ vec3 min(vec3 u, vec3 v);
__builtin__ vec4 min(vec4 u, vec4 v);
__builtin__ vec2 max(vec2 u, vec2 v);
__builtin__ vec3 max(vec3 u, vec3 v);
__builtin__ vec4 max(vec4 u, vec4 v);

__builtin__ ivec2 min(ivec2 u, ivec2 v);
__builtin__ ivec3 min(ivec3 u, ivec3 v);
__builtin__ ivec4 min(ivec4 u, ivec4 v);
__builtin__ ivec2 max(ivec2 u, ivec2 v);
__builtin__ ivec3 max(ivec3 u, ivec3 v);
__builtin__ ivec4 max(ivec4 u, ivec4 v);


__builtin__ uvec2 min(uvec2 u, uvec2 v);
__builtin__ uvec3 min(uvec3 u, uvec3 v);
__builtin__ uvec4 min(uvec4 u, uvec4 v);
__builtin__ uvec2 max(uvec2 u, uvec2 v);
__builtin__ uvec3 max(uvec3 u, uvec3 v);
__builtin__ uvec4 max(uvec4 u, uvec4 v);

__builtin__ float pow(float u, float v);
__builtin__ vec2 pow(vec2 x, vec2 y);
__builtin__ vec3 pow(vec3 x, vec3 y);
__builtin__ vec4 pow(vec4 x, vec4 y);

__builtin__ vec2 clamp(vec2 x, vec2 minVal, vec2 maxVal);
__builtin__ vec3 clamp(vec3 x, vec3 minVal, vec3 maxVal);
__builtin__ vec4 clamp(vec4 x, vec4 minVal, vec4 maxVal);


__builtin__ vec2 clamp(vec2 x, float minVal, float maxVal);
__builtin__ vec3 clamp(vec3 x, float minVal, float maxVal);
__builtin__ vec4 clamp(vec4 x, float minVal, float maxVal);

__builtin__ ivec2 clamp(ivec2 x, int minVal, int maxVal);
__builtin__ ivec3 clamp(ivec3 x, int minVal, int maxVal);
__builtin__ ivec4 clamp(ivec4 x, int minVal, int maxVal);