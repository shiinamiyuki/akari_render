use crate::sampler::ReplaySampler;

use super::mmlt::FRecord;
use crate::sampler::MLTSampler;

pub struct Chain {
    pub sampler: ReplaySampler<MLTSampler>,
    pub cur: FRecord,
}
