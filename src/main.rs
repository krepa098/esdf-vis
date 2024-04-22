use std::collections::BTreeSet;

use integrators::{
    esdf_gpu::{EsdfIntegrator, EsdfIntegratorConfig},
    tsdf::{TsdfIntegrator, TsdfIntegratorConfig},
};

use renderer::Renderer;

mod core;
mod integrators;
mod renderer;
mod wgpu_utils;

type TsdfLayer = core::layer::Layer<core::voxel::Tsdf, 8>;
type EsdfLayer = core::layer::Layer<core::voxel::Esdf, 8>;

fn main() {
    futures::executor::block_on(run());
}

async fn run() {
    let (device, mut queue) = wgpu_utils::create_adapter().await.unwrap();

    let map_img = image::io::Reader::open(format!("{}/maps/map3.png", env!("CARGO_MANIFEST_DIR")))
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();

    let mut tsdf_layer = TsdfLayer::new(1.0);
    let mut tsdf_integrator = TsdfIntegrator::new(TsdfIntegratorConfig::default());

    let mut esdf_layer = EsdfLayer::new(1.0);
    let mut esdf_integrator = EsdfIntegrator::new(EsdfIntegratorConfig::default());

    let renderer = std::rc::Rc::new(std::cell::RefCell::new(Renderer::new(false)));

    // map to tsdf
    let mut dirty_blocks = BTreeSet::new();
    tsdf_integrator.integrate_image(&mut tsdf_layer, &map_img, &mut dirty_blocks);

    // generate esdf and render on callback
    let renderer_cb = renderer.clone();
    esdf_integrator
        .update_blocks(
            &tsdf_layer,
            &mut esdf_layer,
            &dirty_blocks,
            &device,
            &mut queue,
            move |op, tsdf_layer, esdf_layer, block_indices, duration| {
                renderer_cb.borrow_mut().render_tsdf_layer(
                    tsdf_layer,
                    esdf_layer,
                    block_indices,
                    op,
                    Some(duration),
                );
            },
        )
        .await;

    renderer.borrow_mut().render_tsdf_layer(
        &tsdf_layer,
        &esdf_layer,
        &[],
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
    tsdf_integrator.integrate_image(&mut tsdf_layer, &map_img, &mut dirty_blocks);

    // generate esdf and render on callback
    let renderer_cb = renderer.clone();
    esdf_integrator
        .update_blocks(
            &tsdf_layer,
            &mut esdf_layer,
            &dirty_blocks,
            &device,
            &mut queue,
            move |op, tsdf_layer, esdf_layer, block_indices, duration| {
                renderer_cb.borrow_mut().render_tsdf_layer(
                    tsdf_layer,
                    esdf_layer,
                    block_indices,
                    op,
                    Some(duration),
                );
            },
        )
        .await;

    renderer.borrow_mut().render_tsdf_layer(
        &tsdf_layer,
        &esdf_layer,
        &[],
        "",
        Some(std::time::Duration::from_secs(4)),
    );

    renderer
        .borrow_mut()
        .render_gif(&format!("{}/tsdf.gif", env!("CARGO_MANIFEST_DIR")));
}
