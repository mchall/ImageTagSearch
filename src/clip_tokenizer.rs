use anyhow::{Context, Result};
use regex::Regex;
use std::{collections::HashMap, path::Path};

pub struct Tokenizer {
    vocab: HashMap<String, i64>,
    ranks: HashMap<(String, String), usize>,
    pattern: Regex,
    bytes: [char; 256],
}

impl Tokenizer {
    pub fn load(folder: &Path) -> Result<Self> {
        let vocab = serde_json::from_slice(&std::fs::read(folder.join("vocab.json"))?)?;
        let ranks = std::fs::read_to_string(folder.join("merges.txt"))?
            .lines()
            .filter(|s| !s.starts_with('#') && !s.trim().is_empty())
            .filter_map(|s| s.split_once(' '))
            .enumerate()
            .map(|(i, (a, b))| ((a.to_owned(), b.to_owned()), i))
            .collect();
        let mut extra = 0;
        let bytes = std::array::from_fn(|b| {
            let printable =
                (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
            let code = if printable {
                b
            } else {
                extra += 1;
                255 + extra
            };
            char::from_u32(code as u32).unwrap()
        });
        Ok(Self {
            vocab,
            ranks,
            bytes,
            pattern: Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+")?,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let start = *self
            .vocab
            .get("<|startoftext|>")
            .context("Missing CLIP start token")?;
        let end = *self
            .vocab
            .get("<|endoftext|>")
            .context("Missing CLIP end token")?;
        let mut ids = vec![start];
        for word in self.pattern.find_iter(&text.to_lowercase()) {
            let mut parts: Vec<String> = word
                .as_str()
                .bytes()
                .map(|b| self.bytes[b as usize].to_string())
                .collect();
            if let Some(last) = parts.last_mut() {
                last.push_str("</w>");
            }
            while parts.len() > 1 {
                let best = parts
                    .windows(2)
                    .filter_map(|p| {
                        self.ranks
                            .get(&(p[0].clone(), p[1].clone()))
                            .map(|r| (*r, p[0].clone(), p[1].clone()))
                    })
                    .min_by_key(|p| p.0);
                let Some((_, a, b)) = best else { break };
                let mut merged = Vec::new();
                let mut i = 0;
                while i < parts.len() {
                    if i + 1 < parts.len() && parts[i] == a && parts[i + 1] == b {
                        merged.push(format!("{a}{b}"));
                        i += 2;
                    } else {
                        merged.push(parts[i].clone());
                        i += 1;
                    }
                }
                parts = merged;
            }
            for part in parts {
                if ids.len() == 76 {
                    break;
                }
                ids.push(*self.vocab.get(&part).unwrap_or(&end));
            }
            if ids.len() == 76 {
                break;
            }
        }
        ids.push(end);
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_clip_tokens_and_truncation() {
        let t = Tokenizer::load(
            crate::runtime::find_asset("Model/CLIP/vocab.json")
                .parent()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            t.encode("a photo of a cat").unwrap(),
            [49406, 320, 1125, 539, 320, 2368, 49407]
        );
        assert_eq!(
            t.encode("  A PHOTO of a CAT ").unwrap(),
            t.encode("a photo of a cat").unwrap()
        );
        let long = t.encode(&"cat ".repeat(100)).unwrap();
        assert_eq!(long.len(), 77);
        assert_eq!(long[76], 49407);
        assert_eq!(t.encode("").unwrap(), [49406, 49407]);
        assert!(t.encode("café 日本語 123!").unwrap().len() > 5);
    }
}
