#[derive(Default, Debug, Clone)]
pub enum Activation {
    #[default]
    Identity,
    BinaryStep,
    LeakyReLU,
    ReLU,
    Sigmoid,
}

impl Activation {
    pub fn compute(&self, x: f32) -> f32 {
        return match self {
            Self::ReLU => {
                if x < 0. {
                    0.
                } else {
                    x
                }
            }
            Self::Sigmoid => x.exp() / (1. + x.exp()),
            Self::BinaryStep => {
                if x < 0. {
                    0.
                } else {
                    1.
                }
            }
            Self::LeakyReLU => {
                if x < 0. {
                    0.01 * x
                } else {
                    x
                }
            }
            _ => x,
        };
    }
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Self::ReLU => {
                if x < 0. {
                    0.
                } else {
                    1.
                }
            }
            Self::Sigmoid => self.compute(x) * (1.0 - self.compute(x)),
            Self::BinaryStep => 0.,
            Self::LeakyReLU => {
                if x < 0. {
                    0.01
                } else {
                    1.
                }
            }
            _ => 1.,
        }
    }
}
