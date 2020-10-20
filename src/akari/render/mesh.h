#pragma once

#include <akari/render/instance.h>
namespace akari::render {
    class MeshNode : public ShapeNode {
      public:
        virtual MeshInstance create_instance(Allocator<> *) = 0;
    };
} // namespace akari::render