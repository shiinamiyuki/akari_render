#pragma once
namespace akari{
    template<typename Float, typename Spectrum> struct Ray;
    template<typename Float, typename Spectrum> struct Intersection;
    template<typename Float, typename Spectrum> struct Triangle;
    template<typename Float, typename Spectrum> struct ShadingPoint;
    template<typename Float, typename Spectrum> class  BSDF;
    template<typename Float, typename Spectrum> class  BSDFComponent;
    template<typename Float, typename Spectrum> class  Material;
    template<typename Float, typename Spectrum> class  EndPoint;
    template<typename Float, typename Spectrum> class  Camera;
    template<typename Float, typename Spectrum> class  Sampler;
    template<typename Float, typename Spectrum> class  Texture;
    template<typename Float, typename Spectrum> class  Integrator;
    template<typename Float, typename Spectrum> struct SurfaceSample;
    template<typename Float, typename Spectrum> struct Interaction;
    template<typename Float, typename Spectrum> struct SurfaceInteraction;
    template<typename Float, typename Spectrum> struct EndPointInteraction;
    template<typename Float, typename Spectrum> struct VolumeInteraction;
    template<typename Float, typename Spectrum> struct MaterialSlot;
    template<typename Float, typename Spectrum> class  Light;
    template<typename Float, typename Spectrum> class  Mesh;
    template<typename Float, typename Spectrum> struct Emission;
    template<typename Float, typename Spectrum> class  Scene;
    template<typename Float, typename Spectrum> struct BSDFSample;
    template<typename Float, typename Spectrum> struct Collection {
        using Ray = Ray<Float, Spectrum>;
        using Intersection = Intersection<Float, Spectrum>;
        using Triangle = Triangle<Float, Spectrum>;
        using ShadingPoint = ShadingPoint<Float, Spectrum>;
        using BSDF = BSDF<Float, Spectrum>;
        using BSDFComponent = BSDFComponent<Float, Spectrum>;
        using Material = Material<Float, Spectrum>;
        using EndPoint = EndPoint<Float, Spectrum>;
        using Camera = Camera<Float, Spectrum>;
        using Sampler = Sampler<Float, Spectrum>;
        using Texture = Texture<Float, Spectrum>;
        using Integrator = Integrator<Float, Spectrum>;
        using SurfaceSample = SurfaceSample<Float, Spectrum>;
        using Interaction = Interaction<Float, Spectrum>;
        using SurfaceInteraction = SurfaceInteraction<Float, Spectrum>;
        using EndPointInteraction = EndPointInteraction<Float, Spectrum>;
        using VolumeInteraction = VolumeInteraction<Float, Spectrum>;
        using MaterialSlot = MaterialSlot<Float, Spectrum>;
        using Light = Light<Float, Spectrum>;
        using Mesh = Mesh<Float, Spectrum>;
        using Emission = Emission<Float, Spectrum>;
        using Scene = Scene<Float, Spectrum>;
        using BSDFSample = BSDFSample<Float, Spectrum>;
    };

}
