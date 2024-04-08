use core::index::BlockIndex;
use std::collections::BTreeSet;

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
    let mut dirty_blocks = BTreeSet::new();
    tsdf_integrator.integrate_image(&mut tsdf_layer, &map_img, &mut dirty_blocks);

    // generate esdf and render on callback
    let renderer_cb = renderer.clone();
    esdf_integrator.update_blocks(
        &tsdf_layer,
        &mut esdf_layer,
        &dirty_blocks,
        move |op, tsdf_layer, esdf_layer, block_index| {
            renderer_cb.borrow_mut().render_tsdf_layer(
                tsdf_layer,
                esdf_layer,
                Some(block_index),
                op,
                None,
            );
        },
    );

    renderer.borrow_mut().render_tsdf_layer(
        &tsdf_layer,
        &esdf_layer,
        None,
        "",
        Some(std::time::Duration::from_secs(2)),
    );

    let map_img = image::io::Reader::open(format!("{}/maps/map3b.png", env!("CARGO_MANIFEST_DIR")))
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    // map to tsdf
    dirty_blocks.clear();
    // tsdf_layer.clear(); // partial updates not implemented
    tsdf_integrator.integrate_image(&mut tsdf_layer, &map_img, &mut dirty_blocks);

    // generate esdf and render on callback
    // esdf_layer.clear(); // partial updates not implemented
    let renderer_cb = renderer.clone();
    esdf_integrator.update_blocks(
        &tsdf_layer,
        &mut esdf_layer,
        &dirty_blocks,
        move |op, tsdf_layer, esdf_layer, block_index| {
            renderer_cb.borrow_mut().render_tsdf_layer(
                tsdf_layer,
                esdf_layer,
                Some(block_index),
                op,
                None,
            );
        },
    );

    renderer.borrow_mut().render_tsdf_layer(
        &tsdf_layer,
        &esdf_layer,
        None,
        "",
        Some(std::time::Duration::from_secs(2)),
    );

    renderer
        .borrow_mut()
        .render_gif(&format!("{}/tsdf.gif", env!("CARGO_MANIFEST_DIR")));
}
