mod clip;
mod clip_ui;
mod runtime;

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|a| a == "--smoke-test") {
        return clip::smoke_test();
    }
    eframe::run_native(
        "Photo Search · CLIP",
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default()
                .with_inner_size([1360.0, 900.0])
                .with_min_inner_size([850.0, 600.0]),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(clip_ui::App::new(&cc.egui_ctx)))),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))
}
