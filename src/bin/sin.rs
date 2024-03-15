use neural_network_lib::*;
use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut network = Network::default();
    network.add_layer(Layer::new(1, 20, Activation::ReLU));
    network.add_layer(Layer::new(20, 20, Activation::ReLU));
    network.add_layer(Layer::new(20, 1, Activation::Identity));
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    for i in 0..400 {
        let i = i as f32 / 100.;
        let s = i.sin();
        x_train.push(vec![i]);
        y_train.push(vec![s]);
    }
    network.train(x_train, y_train, 1000, 0.01);

    let root = BitMapBackend::new("sin.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("NN prediction - Sin", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..std::f32::consts::PI, 0f32..1.5f32)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        (0..300)
            .map(|x| x as f32 * std::f32::consts::PI / 300.)
            .map(|x| (x, x.sin())),
        &RED,
    ))?;

    chart.draw_series(LineSeries::new(
        (0..300)
            .map(|x| x as f32 * std::f32::consts::PI / 300.)
            .map(|x| (x, network.forward(vec![x])[0])),
        &BLUE,
    ))?;

    root.present()?;

    Ok(())
}
