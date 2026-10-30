#[path = "clip_tokenizer.rs"]
mod tokenizer;
use crate::runtime;
use anyhow::{bail, ensure, Context, Result};

use ort::{session::Session, value::Tensor};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
    },
    time::SystemTime,
};

const SIDE: usize = 224;
const MEAN: [f32; 3] = [0.5; 3];
const STD: [f32; 3] = [0.5; 3];

#[derive(Clone)]
pub struct Hit {
    pub path: PathBuf,
    pub score: f32,
}
pub struct Search {
    pub folder: PathBuf,
    pub models: PathBuf,
    pub query: String,
    pub recursive: bool,
    pub gpu: bool,
    pub adapter: i32,
}
pub enum Event {
    Progress(String),
    Device(String),
    Done {
        hits: Vec<Hit>,
        total: usize,
        skipped: usize,
    },
    Failed(String),
}
struct Cached {
    size: u64,
    modified: Option<SystemTime>,
    vector: Vec<f32>,
}
struct Models {
    text: Session,
    vision: Session,
    tokenizer: tokenizer::Tokenizer,
}
#[derive(Default)]
pub struct Engine {
    models: Option<Models>,
    key: Option<(PathBuf, bool, i32)>,
    images: HashMap<PathBuf, Cached>,
    texts: HashMap<String, Vec<f32>>,
}

fn normalize(values: &[f32]) -> Result<Vec<f32>> {
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "SigLIP2 returned an invalid embedding"
    );
    Ok(values.iter().map(|v| v / norm).collect())
}
fn preprocess(path: &Path) -> Result<Vec<f32>> {
    let image = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(
        &image,
        SIDE as u32,
        SIDE as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut data = vec![0.0; 3 * SIDE * SIDE];
    for (x, y, p) in resized.enumerate_pixels() {
        for c in 0..3 {
            data[c * SIDE * SIDE + y as usize * SIDE + x as usize] =
                (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    Ok(data)
}
fn image_embedding(session: &mut Session, path: &Path) -> Result<Vec<f32>> {
    let input = Tensor::from_array(([1, 3, SIDE, SIDE], preprocess(path)?))?;
    let outputs = session.run(ort::inputs!["pixel_values" => input])?;
    let (shape, values) = outputs
        .get("pooler_output")
        .context("Missing SigLIP2 vision pooler_output")?
        .try_extract_tensor::<f32>()?;
    ensure!(
        shape.len() == 2 && shape[0] == 1,
        "Unexpected SigLIP2 vision output: {shape:?}"
    );
    normalize(values)
}
fn text_embedding(models: &mut Models, query: &str) -> Result<Vec<f32>> {
    let ids = models.tokenizer.encode(query)?;
    let input = Tensor::from_array(([1, ids.len()], ids))?;
    let outputs = models.text.run(ort::inputs!["input_ids" => input])?;
    let (shape, values) = outputs
        .get("pooler_output")
        .context("Missing SigLIP2 text pooler_output")?
        .try_extract_tensor::<f32>()?;
    ensure!(
        shape.len() == 2 && shape[0] == 1,
        "Unexpected SigLIP2 text output: {shape:?}"
    );
    normalize(values)
}
fn cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        bail!("Search cancelled");
    }
    Ok(())
}
fn list(
    folder: &Path,
    recursive: bool,
    cancel: &AtomicBool,
    files: &mut Vec<PathBuf>,
    warnings: &mut usize,
) -> Result<()> {
    for entry in
        std::fs::read_dir(folder).with_context(|| format!("Cannot read {}", folder.display()))?
    {
        cancelled(cancel)?;
        let entry = match entry {
            Ok(e) => e,
            Err(_) => {
                *warnings += 1;
                continue;
            }
        };
        let kind = entry.file_type()?;
        if kind.is_dir() && recursive {
            if let Err(e) = list(&entry.path(), true, cancel, files, warnings) {
                cancelled(cancel)?;
                eprintln!("{e:#}");
                *warnings += 1;
            }
        } else if kind.is_file()
            && entry
                .path()
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| {
                    matches!(
                        s.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp"
                    )
                })
        {
            files.push(entry.path());
        }
    }
    Ok(())
}
impl Engine {
    pub fn search(
        &mut self,
        request: Search,
        tx: &Sender<Event>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        ensure!(!request.query.trim().is_empty(), "Enter some search text");
        let mut files = Vec::new();
        let mut skipped = 0;
        list(
            &request.folder,
            request.recursive,
            cancel,
            &mut files,
            &mut skipped,
        )?;
        files.sort();
        ensure!(
            !files.is_empty(),
            "No supported images found in this folder"
        );
        let key = (request.models.clone(), request.gpu, request.adapter);
        if self.key.as_ref() != Some(&key) {
            let _ = tx.send(Event::Progress(
                "Loading SigLIP2 text and vision models…".into(),
            ));
            let tokenizer = tokenizer::Tokenizer::load(&request.models)?;
            let (text, td) = runtime::create_session(
                &request.models.join("text_model_fp16.onnx"),
                request.gpu,
                request.adapter,
            )?;
            cancelled(cancel)?;
            let (vision, vd) = runtime::create_session(
                &request.models.join("vision_model_fp16.onnx"),
                request.gpu,
                request.adapter,
            )?;
            self.models = Some(Models {
                text,
                vision,
                tokenizer,
            });
            self.key = Some(key);
            self.images.clear();
            self.texts.clear();
            let _ = tx.send(Event::Device(format!("Text: {td} · Images: {vd}")));
        }
        let models = self.models.as_mut().unwrap();
        if !self.texts.contains_key(&request.query) {
            self.texts.insert(
                request.query.clone(),
                text_embedding(models, &request.query)?,
            );
        }
        let query = &self.texts[&request.query];
        let mut hits = Vec::new();
        for (i, path) in files.iter().enumerate() {
            cancelled(cancel)?;
            let _ = tx.send(Event::Progress(format!(
                "Analysing image {} of {}",
                i + 1,
                files.len()
            )));
            let result = (|| -> Result<f32> {
                let meta = std::fs::metadata(path)?;
                let fresh = self
                    .images
                    .get(path)
                    .is_some_and(|c| c.size == meta.len() && c.modified == meta.modified().ok());
                if !fresh {
                    let vector = image_embedding(&mut models.vision, path)
                        .with_context(|| format!("Cannot embed {}", path.display()))?;
                    self.images.insert(
                        path.clone(),
                        Cached {
                            size: meta.len(),
                            modified: meta.modified().ok(),
                            vector,
                        },
                    );
                }
                let vector = &self.images[path].vector;
                ensure!(
                    query.len() == vector.len(),
                    "SigLIP2 text and vision dimensions do not match"
                );
                Ok(query.iter().zip(vector).map(|(a, b)| a * b).sum())
            })();
            match result {
                Ok(score) => hits.push(Hit {
                    path: path.clone(),
                    score,
                }),
                Err(e) => {
                    skipped += 1;
                    eprintln!("{e:#}");
                }
            }
        }
        cancelled(cancel)?;
        ensure!(!hits.is_empty(),"Could not embed any images. Check that the SigLIP2 model pair is compatible; see console for file errors.");
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.path.cmp(&b.path))
        });
        let _ = tx.send(Event::Done {
            hits,
            total: files.len(),
            skipped,
        });
        Ok(())
    }
}

pub fn smoke_test() -> Result<()> {
    runtime::init_runtime()?;
    let folder = std::env::args()
        .skip_while(|s| s != "--folder")
        .nth(1)
        .map(PathBuf::from)
        .context("Usage: clip-search --smoke-test --folder <image folder>")?;
    let (tx, rx) = std::sync::mpsc::channel();
    let mut engine = Engine::default();
    for query in ["a swimming pool", "a bedroom"] {
        let start = std::time::Instant::now();
        engine.search(
            Search {
                folder: folder.clone(),
                models: runtime::find_asset("Model/SigLIP2/text_model_fp16.onnx")
                    .parent()
                    .unwrap()
                    .into(),
                query: query.into(),
                recursive: true,
                gpu: false,
                adapter: 0,
            },
            &tx,
            &AtomicBool::new(false),
        )?;
        while let Ok(event) = rx.try_recv() {
            if let Event::Done {
                hits,
                total,
                skipped,
            } = event
            {
                ensure!(
                    hits.windows(2).all(|w| w[0].score >= w[1].score),
                    "Results not sorted"
                );
                ensure!(
                    hits.iter()
                        .all(|h| h.score.is_finite() && h.score.abs() <= 1.001),
                    "Invalid cosine scores"
                );
                println!(
                    "PASS {query}: {total} images, {skipped} skipped, {:.2}s",
                    start.elapsed().as_secs_f32()
                );
                for hit in hits.iter().take(3) {
                    println!("{:.4} {}", hit.score, hit.path.display());
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn siglip_preprocessing_and_normalization() {
        let path = std::env::temp_dir().join(format!("clip-test-{}.png", std::process::id()));
        image::RgbImage::from_pixel(224, 448, image::Rgb([255, 128, 0]))
            .save(&path)
            .unwrap();
        let data = preprocess(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(data.len(), 3 * SIDE * SIDE);
        for c in 0..3 {
            let expected = ([1.0, 128.0 / 255.0, 0.0][c] - MEAN[c]) / STD[c];
            assert!(data[c * SIDE * SIDE..(c + 1) * SIDE * SIDE]
                .iter()
                .all(|v| (v - expected).abs() < 0.0001));
        }
        assert!(normalize(&[0.0, 0.0]).is_err());
        assert_eq!(normalize(&[3.0, 4.0]).unwrap(), [0.6, 0.8]);
    }

    #[test]
    fn resize_preserves_edges_of_non_square_image() {
        let path = std::env::temp_dir().join(format!("siglip-edges-{}.png", std::process::id()));
        let image = image::RgbImage::from_fn(448, 224, |x, _| {
            if !(56..392).contains(&x) {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            }
        });
        image.save(&path).unwrap();
        let data = preprocess(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        // A CLIP-style center crop would discard both red edges.
        assert_eq!(data[0], 1.0);
        assert_eq!(data[SIDE - 1], 1.0);
        assert_eq!(data[SIDE / 2], -1.0);
    }
}
