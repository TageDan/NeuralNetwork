use rand::random;

use crate::{Activation, Perceptron};

#[derive(Default, Debug, Clone)]
pub struct Layer {
    input_length: usize,
    output_length: usize,
    weights: Vec<Vec<f32>>,
    perceptrons: Vec<Perceptron>,
    activation: Activation,
    weight_gradient: Vec<Vec<f32>>,
    bias_gradient: Vec<f32>,
    last_input: Vec<f32>,
}

impl Layer {
    pub fn new(input: usize, output: usize, activation: Activation) -> Self {
        let mut weights = Vec::new();
        let mut perceptrons = Vec::new();
        for _ in 0..output {
            perceptrons.push(Perceptron::default());
            let mut weigths_for_perceptron = Vec::new();
            for _ in 0..input {
                weigths_for_perceptron.push(random::<f32>() * 2. - 1.);
            }

            weights.push(weigths_for_perceptron);
        }
        Self {
            input_length: input,
            output_length: output,
            weights,
            perceptrons,
            activation,
            weight_gradient: vec![vec![0.; input]; output],
            bias_gradient: vec![0.; output],
            last_input: vec![],
        }
    }

    pub fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut result = Vec::new();
        if input.len() != self.input_length {
            panic!("Dimensions of layers does not match!");
        }
        self.last_input = input.clone();
        for perceptron in 0..self.output_length {
            let mut val = self.perceptrons[perceptron].bias;
            for index in 0..self.input_length {
                val += input[index] * self.weights[perceptron][index];
            }
            self.perceptrons[perceptron].last_weighted_sum = val;
            val = self.activation.compute(val);
            self.perceptrons[perceptron].last_activation = val;
            result.push(val);
        }
        result
    }

    pub fn run(&self, input: Vec<f32>) -> Vec<f32> {
        let mut result = Vec::new();
        if input.len() != self.input_length {
            panic!("Dimensions of layers does not match!");
        }
        for perceptron in 0..self.output_length {
            let mut val = self.perceptrons[perceptron].bias;
            for index in 0..self.input_length {
                val += input[index] * self.weights[perceptron][index];
            }
            val = self.activation.compute(val);
            result.push(val);
        }
        result
    }

    fn cost_derivative(
        next_layer_perceptrons: &Option<Layer>,
        j: usize,
        perceptron: &mut Perceptron,
        target_output: Option<&Vec<f32>>,
    ) {
        if let Some(ref next_layer) = *next_layer_perceptrons {
            let mut sum = 0.;
            for i in 0..next_layer.output_length {
                let next_perceptron = &next_layer.perceptrons[i];
                sum += next_perceptron.cost_derivative
                    * next_perceptron.activation_derivative
                    * next_layer.weights[i][j];
            }
            perceptron.cost_derivative = sum;
        } else {
            if let Some(target) = target_output {
                perceptron.cost_derivative = 2. * (perceptron.last_activation - target[j])
            } else {
                panic!("Target should not be None if next_layer is!");
            }
        }
    }

    pub fn backward(
        &mut self,
        next_layer_perceptrons: Option<Self>,
        target_output: Option<&Vec<f32>>,
    ) {
        let mut weight_gradients: Vec<Vec<f32>> = Vec::new();
        let mut bias_gradients: Vec<f32> = Vec::new();
        for j in 0..self.output_length {
            let mut gradient = Vec::new();
            let perceptron = &mut self.perceptrons[j];

            Self::cost_derivative(&next_layer_perceptrons, j, perceptron, target_output);

            perceptron.activation_derivative =
                self.activation.derivative(perceptron.last_weighted_sum);

            for k in 0..self.input_length {
                gradient.push(
                    self.last_input[k]
                        * perceptron.cost_derivative
                        * perceptron.activation_derivative,
                );
            }
            bias_gradients.push(perceptron.cost_derivative * perceptron.activation_derivative);
            weight_gradients.push(gradient)
        }
        self.update_gradients(weight_gradients, bias_gradients);
    }

    fn update_gradients(&mut self, weight_gradient: Vec<Vec<f32>>, bias_gradient: Vec<f32>) {
        for i in 0..self.output_length {
            for j in 0..self.input_length {
                self.weight_gradient[i][j] += weight_gradient[i][j];
            }
            self.bias_gradient[i] += bias_gradient[i];
        }
    }

    pub fn update_weights(&mut self, rate: f32) {
        for i in 0..self.output_length {
            for j in 0..self.input_length {
                self.weights[i][j] -= self.weight_gradient[i][j] * rate;
                self.weight_gradient[i][j] = 0.;
            }
            self.perceptrons[i].bias -= self.bias_gradient[i] * rate;
            self.bias_gradient[i] = 0.;
        }
    }
}
