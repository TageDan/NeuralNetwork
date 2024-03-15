use rand::random;
#[derive(Debug, Clone)]
pub struct Perceptron {
    pub bias: f32,
    pub last_activation: f32,
    pub last_weighted_sum: f32,
    pub cost_derivative: f32,
    pub activation_derivative: f32,
}

impl Default for Perceptron {
    fn default() -> Self {
        Self {
            bias: random::<f32>() * 2. - 1.,
            last_activation: 0.0,
            last_weighted_sum: 0.0,
            cost_derivative: 0.0,
            activation_derivative: 0.0,
        }
    }
}
