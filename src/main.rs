use integrators::{
    esdf::{EsdfIntegrator, EsdfIntegratorConfig},
    tsdf::{TsdfIntegrator, TsdfIntegratorConfig},
};
use renderer::Renderer;

mod core;
mod integrators;
mod renderer;

type TsdfLayer = core::layer::Layer<core::voxel::Tsdf, 16>;
type EsdfLayer = core::layer::Layer<core::voxel::Esdf, 16>;

fn main() {
    let map_img = image::io::Reader::open(format!("{}/maps/map3.png", env!("CARGO_MANIFEST_DIR")))
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    let mut tsdf_layer = TsdfLayer::new(1.0);
    let mut tsdf_integrator = TsdfIntegrator::new(TsdfIntegratorConfig::default());

    let mut esdf_layer = EsdfLayer::new(1.0);
    let mut esdf_integrator = EsdfIntegrator::new(EsdfIntegratorConfig::default());

    let renderer = std::rc::Rc::new(std::cell::RefCell::new(Renderer::new()));

    // map to tsdf
    tsdf_integrator.integrate_image(&mut tsdf_layer, &map_img);

    // generate esdf and render on callback
    let renderer_cb = renderer.clone();
    esdf_integrator.update_blocks(
        &tsdf_layer,
        &mut esdf_layer,
        move |op, tsdf_layer, esdf_layer, block_index| {
            renderer_cb
                .borrow_mut()
                .render_tsdf_layer(tsdf_layer, esdf_layer, block_index, op);
        },
    );

    let map_img = image::io::Reader::open(format!("{}/maps/map4.png", env!("CARGO_MANIFEST_DIR")))
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    // map to tsdf
    tsdf_layer.clear(); // partial updates not implemented
    tsdf_integrator.integrate_image(&mut tsdf_layer, &map_img);

    // generate esdf and render on callback
    esdf_layer.clear(); // partial updates not implemented
    let renderer_cb = renderer.clone();
    esdf_integrator.update_blocks(
        &tsdf_layer,
        &mut esdf_layer,
        move |op, tsdf_layer, esdf_layer, block_index| {
            renderer_cb
                .borrow_mut()
                .render_tsdf_layer(tsdf_layer, esdf_layer, block_index, op);
        },
    );

    renderer
        .borrow_mut()
        .render_gif(&format!("{}/tsdf.gif", env!("CARGO_MANIFEST_DIR")));
}
