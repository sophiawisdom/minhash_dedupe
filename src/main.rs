use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Deserializer;
use std::io::BufRead;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Write;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Arc;
use tiktoken_rs::cl100k_base;

#[derive(Serialize, Deserialize, Debug)]
pub struct File {
    pub filename: String,
    pub contents: String,
}

mod rensa;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        println!("Gotta give me one arg, a filename!! you gave {:?}", args);
        return;
    }
    let filename = args[1].clone();
    println!("Operating with filename {}", filename);
    let file = std::fs::File::open(filename).unwrap();
    let progress_bar = ProgressBar::new(file.metadata().unwrap().len());
    progress_bar.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
        .unwrap()
        .progress_chars("#>-"));
    let buf_reader = BufReader::with_capacity(128 * 1024, file);

    // copied params from The Stack
    let big_lsh = Arc::new(parking_lot::RwLock::new(rensa::RMinHashLSH::new(
        0.7, 256, 5,
    )));

    let accepted_tokens = Arc::new(AtomicU64::new(0));
    let accepted_bytes = Arc::new(AtomicU64::new(0));

    let enc = cl100k_base().unwrap();

    let accepted = Arc::new(AtomicU64::new(0));
    let content_bytes = Arc::new(AtomicU64::new(0));

    let out_file = Arc::new(parking_lot::Mutex::new(BufWriter::with_capacity(
        1024 * 1024,
        std::fs::File::create("fuzzy_deduped_full_out.jsonl").unwrap(),
    )));

    buf_reader.lines()
            .enumerate()
            .par_bridge()
            .for_each(|(line_idx, line_result)| {
                let line = match line_result {
                    Ok(val) => val,
                    Err(e) => {
                        eprintln!("Got error decoding line: {}", e);
                        return;
                    }
                };
                if line_idx % 10000 == 0 {
                    progress_bar.set_message(format!(
                        "At line {}, {} were accepted. {:.2}B accepted tok / {:.2}GB accepted byte ({:.2}GB total byte)",
                        line_idx,
                        accepted.load(SeqCst),
                        (accepted_tokens.load(SeqCst) as f64) / (1_000_000_000 as f64),
                        (accepted_bytes.load(SeqCst) as f64) / ((1024 * 1024 * 1024) as f64),
                        (content_bytes.load(SeqCst) as f64) / ((1024 * 1024 * 1024) as f64),
                    ));
                }
                progress_bar.inc(line.len() as u64 + 1); // 1 for newline

                for file_with_contents in Deserializer::from_str(&line).into_iter() {
                    let file_with_contents: File = match file_with_contents {
                        Ok(val) => val,
                        Err(e) => {
                            println!("error!");
                            continue;
                        }
                    };
                    content_bytes.fetch_add(file_with_contents.contents.len() as u64, SeqCst);
                    let contents = file_with_contents.contents.to_lowercase();

                    let mut minhasher = rensa::RMinHash::new(256, 0);
                    minhasher.update(contents.split(' ').collect());

                    let mut lsh = big_lsh.write();
                    match lsh.any_matches(&minhasher) {
                        Some(_matches) => {
                            // if it matches previous code, throw it away
                        }
                        None => {
                            // if it matches nothing, accept it as data
                            let current_accepted = accepted.fetch_add(1, SeqCst);
                            lsh.insert(current_accepted as usize, &minhasher);
                            drop(lsh); // Release the write lock

                            accepted_bytes.fetch_add(file_with_contents.contents.len() as u64, SeqCst);
                            accepted_tokens.fetch_add(enc.encode_ordinary(&file_with_contents.contents).len() as u64, SeqCst);
                            if let Ok(serialized) = serde_json::to_string(&file_with_contents) {
                                writeln!(out_file.lock(), "{}", serialized).unwrap();
                            }
                        }
                    }
                }
            });
}
