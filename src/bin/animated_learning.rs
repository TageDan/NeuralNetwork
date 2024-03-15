use nannou::prelude::*;
use neural_network_lib::*;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    _window: window::Id,
    network: Network,
    train_x: Vec<Vec<f32>>,
    train_y: Vec<Vec<f32>>,
}

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(1024, 1024)
        .view(view)
        .build()
        .unwrap();
    let network = Network::from(vec![
        Layer::new(1, 10, Activation::Sigmoid),
        Layer::new(10, 5, Activation::Sigmoid),
        Layer::new(5, 1, Activation::Identity),
    ]);
    let mut train_x = Vec::new();
    let mut train_y = Vec::new();
    for i in -500..500 {
        let x = i as f32 * std::f32::consts::PI / 250.;
        let y = x.powi(2);
        train_x.push(vec![x]);
        train_y.push(vec![y]);
    }
    Model {
        _window,
        network,
        train_x,
        train_y,
    }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    model
        .network
        .train(model.train_x.clone(), model.train_y.clone(), 50, 0.005);
}

fn view(app: &App, model: &Model, frame: Frame) {
    let draw = app.draw();

    draw.background().color(BLACK);
    let network_points = (0..=50).map(|i| {
        let x = i as f32 / 10. - 2.5;
        let point = pt2(x, model.network.run(vec![x])[0]) * 50.;
        (point, RED)
    });

    let points = (0..=50).map(|i| {
        let x = i as f32 / 10. - 2.5;
        let point = pt2(x, x.powi(2)) * 50.;
        (point, STEELBLUE)
    });
    draw.polyline().weight(3.0).points_colored(points);
    draw.polyline().weight(3.0).points_colored(network_points);
    draw.to_frame(app, &frame).unwrap();
}
