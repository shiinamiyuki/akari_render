#pragma once
#include <glm/glm.hpp>
namespace akari::shader {
    struct Light {
        glm::vec3 pos;
        glm::vec3 color;
    };
    inline void swap(float & x, float & y)
    {
        float t = x;
        x = y;
        y = t;
    }
    inline glm::vec3 mul(glm::mat3x3 m, glm::vec3 v)
    {
        return transpose(m) * v;
    }
}
