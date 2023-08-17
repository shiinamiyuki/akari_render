pub trait Filter {
    fn radius(&self) -> f32;
    fn evaluate(&self, x: f32) -> f32;
}
