use super::diffuse::DiffuseBsdf;
use super::{fr_dielectric_integral, SurfaceShader};
use std::rc::Rc;

use crate::color::Color;
use crate::geometry::Frame;
use crate::microfacet::TrowbridgeReitzDistribution;
use crate::sampling::weighted_discrete_choice2_and_remap;
use crate::svm::surface::{fr_dielectric, FresnelDielectric, MicrofacetReflection, Surface};
use crate::svm::{SvmMetalBsdf};
use crate::*;

impl SurfaceShader for SvmMetalBsdf {
    fn closure(&self, svm_eval: &svm::eval::SvmEvaluator<'_>) -> Rc<dyn Surface> {
        todo!()
    }
}