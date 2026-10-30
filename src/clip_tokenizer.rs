use anyhow::{anyhow, Result};
use std::path::Path;
use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

pub struct Tokenizer(tokenizers::Tokenizer);
impl Tokenizer {
    pub fn load(folder: &Path) -> Result<Self> {
        let mut t = tokenizers::Tokenizer::from_file(folder.join("tokenizer.json"))
            .map_err(|e| anyhow!("Cannot load SigLIP2 tokenizer: {e}"))?;
        t.with_truncation(Some(TruncationParams {
            max_length: 64,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("Cannot configure tokenizer: {e}"))?;
        t.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(64),
            pad_id: 0,
            pad_token: "<pad>".into(),
            ..Default::default()
        }));
        Ok(Self(t))
    }
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let text = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        let encoded = self
            .0
            .encode(text, true)
            .map_err(|e| anyhow!("Cannot tokenize query: {e}"))?;
        Ok(encoded.get_ids().iter().map(|&id| i64::from(id)).collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn siglip_padding_eos_and_truncation() {
        let t =
            Tokenizer::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Model/SigLIP2")).unwrap();
        let ids = t.encode("a photo of a cat").unwrap();
        assert_eq!(ids.len(), 64);
        let end = ids.iter().position(|&id| id == 1).unwrap();
        assert!(end > 0);
        assert!(ids[end + 1..].iter().all(|&id| id == 0));
        assert_eq!(ids, t.encode("  A PHOTO of a CAT ").unwrap());
        let long = t.encode(&"cat ".repeat(100)).unwrap();
        assert_eq!(long.len(), 64);
        assert_eq!(long[63], 1);
        assert_eq!(t.encode("").unwrap()[0], 1);
        assert_eq!(t.encode("café 日本語 123!").unwrap().len(), 64);
    }
}
