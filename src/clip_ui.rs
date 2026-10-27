use crate::{
    clip::{Engine, Event, Hit, Search},
    runtime,
};
use eframe::egui::{self, Color32, TextureHandle};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    time::{Duration, Instant},
};

type ImageKey = (PathBuf, bool);
type ImageResult = (ImageKey, Result<egui::ColorImage, String>);
const RESULTS_PER_PAGE: usize = 50;
pub struct App {
    folder: String,
    models: String,
    query: String,
    searched: String,
    recursive: bool,
    gpu: bool,
    adapter: i32,
    hits: Vec<Hit>,
    visible_hits: usize,
    reset_results_scroll: bool,
    selected: usize,
    busy: bool,
    status: String,
    error: Option<String>,
    device: String,
    commands: Sender<Search>,
    events: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    started: Option<Instant>,
    elapsed: Duration,
    image_requests: Sender<ImageKey>,
    image_results: Receiver<ImageResult>,
    pending: HashSet<ImageKey>,
    textures: HashMap<ImageKey, Result<TextureHandle, String>>,
}
impl App {
    pub fn new(ctx: &egui::Context) -> Self {
        ctx.set_visuals(egui::Visuals::light());
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = egui::vec2(10.0, 10.0);
        style.visuals.selection.bg_fill = Color32::from_rgb(99, 76, 230);
        style.visuals.extreme_bg_color = Color32::from_rgb(247, 247, 251);
        style.visuals.widgets.inactive.bg_stroke =
            egui::Stroke::new(1.0_f32, Color32::from_gray(200));
        ctx.set_style(style);
        let (commands, rx) = mpsc::channel();
        let (tx, events) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let repaint = ctx.clone();
        std::thread::spawn(move || {
            let mut engine = Engine::default();
            let mut initialized = false;
            while let Ok(request) = rx.recv() {
                let result = (|| {
                    if !initialized {
                        runtime::init_runtime()?;
                        initialized = true;
                    }
                    engine.search(request, &tx, &worker_cancel)
                })();
                if let Err(e) = result {
                    let _ = tx.send(Event::Failed(format!("{e:#}")));
                }
                repaint.request_repaint();
            }
        });
        let (image_requests, requests) = mpsc::channel::<ImageKey>();
        let (images, image_results) = mpsc::channel();
        let repaint = ctx.clone();
        std::thread::spawn(move || {
            while let Ok((path, large)) = requests.recv() {
                let result = (|| {
                    let img = image::open(&path).map_err(|e| e.to_string())?;
                    let img = img
                        .thumbnail(
                            if large { 1400 } else { 320 },
                            if large { 1000 } else { 240 },
                        )
                        .to_rgba8();
                    Ok(egui::ColorImage::from_rgba_unmultiplied(
                        [img.width() as usize, img.height() as usize],
                        img.as_raw(),
                    ))
                })();
                if images.send(((path, large), result)).is_err() {
                    break;
                }
                repaint.request_repaint();
            }
        });
        Self {
            folder: directories::UserDirs::new()
                .and_then(|d| d.picture_dir().map(|p| p.to_string_lossy().into_owned()))
                .unwrap_or_default(),
            models: runtime::find_asset("Model/CLIP/text_model.onnx")
                .parent()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            query: String::new(),
            searched: String::new(),
            recursive: true,
            gpu: true,
            adapter: 0,
            hits: vec![],
            visible_hits: 0,
            reset_results_scroll: false,
            selected: 0,
            busy: false,
            status: "Choose a photo folder and describe what you’re looking for.".into(),
            error: None,
            device: "Models not loaded".into(),
            commands,
            events,
            cancel,
            started: None,
            elapsed: Duration::ZERO,
            image_requests,
            image_results,
            pending: HashSet::new(),
            textures: HashMap::new(),
        }
    }
    fn search(&mut self) {
        if !std::path::Path::new(&self.folder).is_dir() {
            self.error = Some("Select an existing image folder.".into());
            return;
        }
        if self.query.trim().is_empty() {
            self.error = Some("Enter a description, for example: a swimming pool.".into());
            return;
        }
        for name in [
            "text_model.onnx",
            "vision_model.onnx",
            "vocab.json",
            "merges.txt",
        ] {
            if !std::path::Path::new(&self.models).join(name).is_file() {
                self.error = Some(format!(
                    "Missing {name}. Select the CLIP model folder in Model settings."
                ));
                return;
            }
        }
        self.cancel.store(false, Ordering::Relaxed);
        self.error = None;
        self.busy = true;
        self.started = Some(Instant::now());
        self.searched = self.query.trim().into();
        self.hits.clear();
        self.visible_hits = 0;
        self.reset_results_scroll = true;
        self.textures.clear();
        self.selected = 0;
        self.status = "Finding photos…".into();
        if self
            .commands
            .send(Search {
                folder: self.folder.clone().into(),
                models: self.models.clone().into(),
                query: self.searched.clone(),
                recursive: self.recursive,
                gpu: self.gpu,
                adapter: self.adapter,
            })
            .is_err()
        {
            self.finish();
            self.error = Some("Search worker stopped. Restart the app.".into());
        }
    }
    fn finish(&mut self) {
        self.busy = false;
        if let Some(t) = self.started.take() {
            self.elapsed = t.elapsed();
        }
    }
    fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(e) = self.events.try_recv() {
            match e {
                Event::Progress(s) => self.status = s,
                Event::Device(d) => self.device = d,
                Event::Failed(e) => {
                    self.status = "Search stopped.".into();
                    self.error = Some(e);
                    self.finish();
                }
                Event::Done {
                    hits,
                    total,
                    skipped,
                } => {
                    self.status=format!("{} matches from {total} images for “{}” · {skipped} unreadable files or folders skipped",hits.len(),self.searched);
                    self.visible_hits = hits.len().min(RESULTS_PER_PAGE);
                    self.hits = hits;
                    self.finish();
                }
            }
        }
        while let Ok((key, result)) = self.image_results.try_recv() {
            self.pending.remove(&key);
            // Only retain thumbnails belonging to the current result set.
            if self.hits.iter().any(|h| h.path == key.0) {
                let texture = result.map(|img| {
                    ctx.load_texture(
                        format!("{}-{}", key.0.display(), key.1),
                        img,
                        egui::TextureOptions::LINEAR,
                    )
                });
                self.textures.insert(key, texture);
            }
        }
    }
    fn texture(
        &mut self,
        path: &std::path::Path,
        large: bool,
    ) -> Option<Result<TextureHandle, String>> {
        let key = (path.to_path_buf(), large);
        if let Some(t) = self.textures.get(&key) {
            return Some(t.clone());
        }
        if self.pending.insert(key.clone()) {
            let _ = self.image_requests.send(key);
        }
        None
    }
    fn results(&mut self, ui: &mut egui::Ui) {
        let width = ui.available_width();
        let grid_width = (width * 0.59).floor();
        let preview_width = (width - grid_width - 24.0).max(150.0);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(
                egui::vec2(grid_width, ui.available_height()),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    let cols = ((grid_width / 170.0) as usize).clamp(2, 5);
                    let cell = (grid_width - ((cols - 1) as f32 * 10.0)) / cols as f32;
                    let rows = self.visible_hits.div_ceil(cols);
                    let mut scroll = egui::ScrollArea::vertical()
                        .id_salt("results")
                        .auto_shrink([false, false]);
                    if std::mem::take(&mut self.reset_results_scroll) {
                        scroll = scroll.vertical_scroll_offset(0.0);
                    }
                    scroll.show_rows(ui, cell * 0.8 + 48.0, rows, |ui, range| {
                            for row in range {
                                ui.horizontal_top(|ui| {
                                    for index in row * cols..((row + 1) * cols).min(self.visible_hits)
                                    {
                                        let hit = self.hits[index].clone();
                                        ui.allocate_ui_with_layout(
                                            egui::vec2(cell, cell * 0.8 + 38.0),
                                            egui::Layout::top_down(egui::Align::Min),
                                            |ui| {
                                                let (rect, response) = ui.allocate_exact_size(
                                                    egui::vec2(cell, cell * 0.8),
                                                    egui::Sense::click(),
                                                );
                                                ui.painter().rect_filled(
                                                    rect,
                                                    5.0,
                                                    Color32::from_gray(235),
                                                );
                                                match self.texture(&hit.path, false) {
                                                    Some(Ok(t)) => {
                                                        let size = t.size_vec2();
                                                        let scale = (rect.width() / size.x)
                                                            .min(rect.height() / size.y);
                                                        let dst = egui::Rect::from_center_size(
                                                            rect.center(),
                                                            size * scale,
                                                        );
                                                        ui.painter().image(
                                                            t.id(),
                                                            dst,
                                                            egui::Rect::from_min_max(
                                                                egui::Pos2::ZERO,
                                                                egui::pos2(1.0, 1.0),
                                                            ),
                                                            Color32::WHITE,
                                                        );
                                                    }
                                                    Some(Err(_)) => {
                                                        ui.painter().text(
                                                            rect.center(),
                                                            egui::Align2::CENTER_CENTER,
                                                            "Preview unavailable",
                                                            egui::FontId::proportional(11.0),
                                                            Color32::GRAY,
                                                        );
                                                    }
                                                    None => {
                                                        ui.painter().text(
                                                            rect.center(),
                                                            egui::Align2::CENTER_CENTER,
                                                            "Loading…",
                                                            egui::FontId::proportional(12.0),
                                                            Color32::GRAY,
                                                        );
                                                    }
                                                }
                                                if self.selected == index {
                                                    ui.painter().rect_stroke(
                                                        rect,
                                                        5.0,
                                                        egui::Stroke::new(
                                                            3.0_f32,
                                                            Color32::from_rgb(99, 76, 230),
                                                        ),
                                                        egui::StrokeKind::Inside,
                                                    );
                                                }
                                                if response
                                                    .on_hover_text(hit.path.display().to_string())
                                                    .clicked()
                                                {
                                                    self.selected = index;
                                                }
                                                ui.small(format!("{:.3}", hit.score));
                                                ui.add(
                                                    egui::Label::new(
                                                        hit.path
                                                            .file_name()
                                                            .unwrap_or_default()
                                                            .to_string_lossy(),
                                                    )
                                                    .truncate(),
                                                );
                                            },
                                        );
                                    }
                                });
                            }
                        });
                },
            );
            ui.separator();
            ui.allocate_ui_with_layout(
                egui::vec2(preview_width, ui.available_height()),
                egui::Layout::top_down(egui::Align::Min),
                |ui| {
                    let hit = self.hits[self.selected].clone();
                    ui.label(
                        egui::RichText::new(
                            hit.path.file_name().unwrap_or_default().to_string_lossy(),
                        )
                        .strong(),
                    );
                    ui.small(format!(
                        "Result {} · Similarity {:.3}",
                        self.selected + 1,
                        hit.score
                    ));
                    match self.texture(&hit.path, true) {
                        Some(Ok(t)) => {
                            ui.add(egui::Image::new(&t).max_size(egui::vec2(
                                preview_width,
                                (ui.available_height() - 110.0).max(100.0),
                            )));
                        }
                        Some(Err(e)) => {
                            ui.colored_label(Color32::DARK_RED, e);
                        }
                        None => {
                            ui.spinner();
                        }
                    }
                    ui.add(egui::Label::new(hit.path.display().to_string()).wrap());
                    ui.small("Scores rank matches; they are not probabilities.");
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(self.selected > 0, egui::Button::new("< Previous"))
                            .clicked()
                        {
                            self.selected -= 1;
                        }
                        if ui
                            .add_enabled(
                                self.selected + 1 < self.visible_hits,
                                egui::Button::new("Next >"),
                            )
                            .clicked()
                        {
                            self.selected += 1;
                        }
                    });
                },
            );
        });
    }
}
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll(ctx);
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::WHITE).inner_margin(24.0)).show(ctx,|ui| {
            ui.heading(egui::RichText::new("Photo Search").size(30.0));
            ui.label("Find photos by describing them. CLIP searches the contents of your images locally.");ui.add_space(12.0);
            let mut submit=false;
            ui.add_enabled_ui(!self.busy,|ui| {
                ui.horizontal(|ui| {
                    ui.label("Folder");
                    ui.add(egui::TextEdit::singleline(&mut self.folder).desired_width((ui.available_width()-105.0).max(100.0)));
                    if ui.button("Browse…").clicked() {if let Some(p)=rfd::FileDialog::new().set_directory(&self.folder).pick_folder() {self.folder=p.to_string_lossy().into_owned();}}
                });
                ui.horizontal(|ui| {
                    let input=ui.add_sized([ui.available_width()-120.0,36.0],egui::TextEdit::singleline(&mut self.query).hint_text("Search for… a swimming pool, a sunset, a dog on a beach"));
                    submit=input.lost_focus() && ui.input(|i|i.key_pressed(egui::Key::Enter));
                    submit |= ui.add_sized([110.0,36.0],egui::Button::new(egui::RichText::new("Search").color(Color32::WHITE)).fill(Color32::from_rgb(99,76,230))).clicked();
                });
                ui.horizontal(|ui| {ui.checkbox(&mut self.recursive,"Include subfolders");ui.checkbox(&mut self.gpu,"Use GPU (DirectML)");});
                ui.collapsing("Model settings",|ui| {
                    ui.label("CLIP folder containing text_model.onnx, vision_model.onnx, vocab.json and merges.txt");
                    ui.horizontal(|ui| {ui.add(egui::TextEdit::singleline(&mut self.models).desired_width((ui.available_width()-110.0).max(100.0)));if ui.button("Browse models…").clicked() {if let Some(p)=rfd::FileDialog::new().pick_folder() {self.models=p.to_string_lossy().into_owned();}}});
                    ui.horizontal(|ui| {ui.label("GPU adapter");ui.add(egui::DragValue::new(&mut self.adapter).range(0..=32));});
                    ui.small(&self.device);
                });
            });
            if submit {self.search();}
            ui.horizontal_wrapped(|ui| {
                if self.busy {ui.spinner();if ui.button("Cancel").clicked() {self.cancel.store(true,Ordering::Relaxed);}}
                ui.label(&self.status);
                if self.started.is_some() || self.elapsed!=Duration::ZERO {ui.small(format!("{:.1} s",self.started.map(|t|t.elapsed()).unwrap_or(self.elapsed).as_secs_f32()));}
            });
            if let Some(e)=&self.error {ui.colored_label(Color32::DARK_RED,e);}
            if !self.hits.is_empty() {
                ui.horizontal(|ui| {
                    ui.label(format!("Showing {} of {} matches", self.visible_hits, self.hits.len()));
                    let remaining = self.hits.len() - self.visible_hits;
                    if remaining > 0 && ui.button(format!("Fetch next {}", remaining.min(RESULTS_PER_PAGE))).clicked() {
                        self.visible_hits += remaining.min(RESULTS_PER_PAGE);
                    }
                });
            }
            ui.separator();ui.add_space(8.0);
            if !self.hits.is_empty() {self.results(ui);} else if !self.busy {ui.add_space(70.0);ui.vertical_centered(|ui| {ui.heading("Your photos, found by meaning");ui.label("Choose a folder above, then try “a swimming pool” or “a bright living room”.");ui.small("JPEG · PNG · WebP · GIF · BMP   /   Browse matches in batches of 50");});}
        });
    }
}
impl Drop for App {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
