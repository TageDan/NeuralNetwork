use neural_network_lib::*;

fn main() {
    let mut network = Network::default();
    network.add_layer(Layer::new(2, 10, Activation::LeakyReLU));
    network.add_layer(Layer::new(10, 1, Activation::Sigmoid));
    let train_x = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
    let train_y = vec![vec![0.], vec![1.], vec![1.], vec![0.]];

    println!("before training: ");
    println!("{:?}, answer: {}", network.forward(vec![1., 1.]), 0);
    println!("{:?}, answer: {}", network.forward(vec![0., 1.]), 1);
    println!("{:?}, answer: {}", network.forward(vec![0.9, 0.]), 1);
    println!("{:?}, answer: {}", network.forward(vec![0., 0.]), 0);
    network.train(train_x, train_y, 100000, 0.01);

    println!("after training: ");
    println!("{:?}, answer: {}", network.forward(vec![1., 1.]), 0);
    println!("{:?}, answer: {}", network.forward(vec![0., 1.]), 1);
    println!("{:?}, answer: {}", network.forward(vec![0.9, 0.]), 1);
    println!("{:?}, answer: {}", network.forward(vec![0., 0.]), 0);
}
