#ifndef TEXTURE_EVAL_GLSL
#define TEXTURE_EVAL_GLSL

vec3 evaluate_texture(const Texture tex, vec2 texcoords){
    int type = tex.type;
    texcoords.y = 1.0 - texcoords.y;
    if(type == TEXTURE_FLOAT){
        return vec3(tex.data.x);
    }else if(type == TEXTURE_SPECTRUM){
        return tex.data.xyz;
    }else if(type == TEXTURE_FLOAT_IMAGE){
        return texture(sampler2D(image_textures[tex.image_tex_id], tex_sampler), texcoords).xyz;
    }else if(type == TEXTURE_SPECTRUM_IMAGE){
        return texture(sampler2D(image_textures[tex.image_tex_id], tex_sampler), texcoords).xyz;
    }else{
        return vec3(0,1,0);
    }
}

#endif