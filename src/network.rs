use crate::Layer;

#[derive(Default, Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn from(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, input: Vec<f32>) -> Vec<f32> {
        let mut result = input;
        for i in 0..self.layers.len() {
            result = self.layers[i].forward(result);
        }
        result
    }

    pub fn run(&self, input: Vec<f32>) -> Vec<f32> {
        let mut result = input;
        for i in 0..self.layers.len() {
            result = self.layers[i].run(result);
        }
        result
    }

    pub fn backward(&mut self, target: &Vec<f32>) {
        let last = self.layers.len() - 1;
        self.layers[last].backward(None, Some(target));

        for l in 0..last {
            let next_layer = self.layers[last - l].clone();
            self.layers[last - 1 - l].backward(Some(next_layer), None);
        }
    }

    fn update_layers(&mut self, rate: f32) {
        for layer in self.layers.iter_mut() {
            layer.update_weights(rate);
        }
    }

    pub fn train(
        &mut self,
        input_train: Vec<Vec<f32>>,
        target_train: Vec<Vec<f32>>,
        epoch: u32,
        rate: f32,
    ) {
        for _ in 0..epoch {
            for train in 0..input_train.len() {
                self.forward(input_train[train].clone());
                self.backward(&target_train[train]);
            }
            self.update_layers(rate / (input_train.len() as f32));
        }
    }
}
