# GPU Vanity Address Finder - Deployment Guide

## Server Specs
- **4x RTX 5090**: Instance ID 28624520 ($1.682/hr)
- **8x RTX 5090**: Instance ID 28627637 ($3.634/hr)
- Both at: 103.109.13.158

## Expected Performance
- **Current (CPU-only)**: ~8.5M keys/sec
- **Expected (4x RTX 5090)**: **200M-800M keys/sec** (23-94x faster)
- **Expected (8x RTX 5090)**: **400M-1.6B keys/sec** (47-188x faster)

## Installation on GPU Server

### 1. Install CUDA Toolkit (if not already installed)
```bash
# Check if CUDA is installed
nvidia-smi
nvcc --version

# If not installed, install CUDA 12.x
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run
```

### 2. Install Rust (if not already installed)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup default stable
```

### 3. Clone and Build
```bash
# Upload your code to the server or clone from git
cd /root  # or your preferred directory

# Build with optimizations
cargo build --release

# This will compile the CUDA kernels and Rust code
# Build time: 5-15 minutes depending on server
```

### 4. Run Initial Benchmark
```bash
# Test GPU performance
./target/release/vanity-grinder benchmark

# This will show you the keys/sec rate on your specific GPU setup
```

## Usage Examples

### Basic Pattern Search (Prefix)
```bash
# Search for addresses starting with "wif"
./target/release/vanity-grinder generate wif

# Case-sensitive search
./target/release/vanity-grinder generate WIF --no-case-sensitive=false
```

### Suffix Search
```bash
# Search for addresses ending with "pump"
./target/release/vanity-grinder generate pump --suffix
```

### Multiple Patterns (Your Use Case)
The current tool searches for ONE pattern at a time. For your multiple patterns (wif, mlg, pop, aura) all ending with "pump", you have two options:

**Option 1: Run multiple instances (recommended for different prefixes)**
```bash
# Terminal 1 - Search for "wif...pump"
./target/release/vanity-grinder generate wif --suffix pump &

# Terminal 2 - Search for "mlg...pump"  
./target/release/vanity-grinder generate mlg --suffix pump &

# Terminal 3 - Search for "pop...pump"
./target/release/vanity-grinder generate pop --suffix pump &

# Terminal 4 - Search for "aura...pump"
./target/release/vanity-grinder generate aura --suffix pump &
```

**Option 2: Custom wrapper script (see run_multi_pattern.sh)**

### Time Estimates
```bash
# Estimate time for 4-character pattern
./target/release/vanity-grinder estimate 4

# Estimate time for 6-character pattern
./target/release/vanity-grinder estimate 6
```

## Multi-GPU Utilization

The tool automatically uses ALL available GPUs. To verify:
```bash
# Check GPU utilization while running
watch -n 1 nvidia-smi

# You should see all GPUs at ~100% utilization
```

To limit to specific GPUs:
```bash
# Use only GPU 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ./target/release/vanity-grinder generate wif
```

## Output

Found keypairs are saved to:
```
keys/prefix_wif_20251208_143022_5aBcDeFg.json
```

Format: `[mode]_[pattern]_[timestamp]_[first8chars].json`

## Troubleshooting

### Error: "CUDA initialization failed"
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify GPU compute capability (need 6.0+)
nvidia-smi --query-gpu=compute_cap --format=csv
```

### Error: "Out of memory"
```bash
# Reduce batch size by editing src/cuda/vanity_generator.rs
# Or limit GPU usage
CUDA_VISIBLE_DEVICES=0 ./target/release/vanity-grinder generate wif
```

### Slow Performance
```bash
# Run benchmark to find optimal batch size
./target/release/vanity-grinder benchmark

# Check GPU utilization
nvidia-smi

# Ensure GPU isn't throttling due to temperature
nvidia-smi -q -d TEMPERATURE
```

## Cost Analysis

### Current CPU Setup
- Speed: 8.5M keys/sec
- To find "wif" + "pump" (7 chars): ~24 hours
- Cost: varies by server

### With 4x RTX 5090 (conservative estimate: 400M keys/sec)
- Speed: 400M keys/sec (47x faster)
- To find "wif" + "pump": ~30 minutes
- Cost: $1.682/hr × 0.5hr = **$0.84 per match**

### With 8x RTX 5090 (conservative estimate: 800M keys/sec)
- Speed: 800M keys/sec (94x faster)
- To find "wif" + "pump": ~15 minutes
- Cost: $3.634/hr × 0.25hr = **$0.91 per match**

**Recommendation**: Use the 4x RTX 5090 server - better cost efficiency for most patterns.
