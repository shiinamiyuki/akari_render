struct LCGSampler {
    uint seed;
}
extension LCGSampler {
    void set_sample_idx(inout Self self, uint idx){
        self.seed = idx;
    }
    float next1d(inout Self self){
        self.seed = (uint(1103515245) *  self.seed + uint(12345));
        return float(self.seed) / float(4294967295);
    }
}