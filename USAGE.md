# Usage Guide

## Your Specific Use Case

Search for addresses that start with "wif", "mlg", "pop", or "aura" AND end with "pump" (all case-insensitive):

```bash
./run_search.sh
```

OR manually:

```bash
./target/release/vanity-grinder multi-prefix \
    --prefixes "wif,mlg,pop,aura" \
    --suffix "pump" \
    --no-case-sensitive
```

## Other Usage Examples

### 1. Single Prefix + Suffix
Search for addresses starting with "wif" AND ending with "pump":

```bash
./target/release/vanity-grinder generate-prefix-suffix wif pump --no-case-sensitive
```

### 2. Prefix Only (Fast!)
Search for addresses starting with "wif":

```bash
./target/release/vanity-grinder generate wif
```

### 3. Suffix Only
Search for addresses ending with "pump":

```bash
./target/release/vanity-grinder generate pump --suffix
```

### 4. Case-Sensitive Search
By default, searches are case-sensitive. To make them case-insensitive, add `--no-case-sensitive`:

```bash
# Case-sensitive (WIF != wif)
./target/release/vanity-grinder generate-prefix-suffix WIF PUMP

# Case-insensitive (WIF = wif = Wif)
./target/release/vanity-grinder generate-prefix-suffix WIF PUMP --no-case-sensitive
```

### 5. Benchmark Your GPU
See how fast your specific GPU setup is:

```bash
./target/release/vanity-grinder benchmark
```

### 6. Estimate Search Time
Estimate how long it will take to find a pattern:

```bash
# For a 4-character pattern (case-sensitive)
./target/release/vanity-grinder estimate 4

# For a 4-character pattern (case-insensitive)
./target/release/vanity-grinder estimate 4 --case-sensitive=false
```

## Expected Search Times

Based on 400M keys/sec (conservative estimate for 4x RTX 5090):

| Pattern | Length | Probability | Estimated Time |
|---------|--------|-------------|----------------|
| wif | 3 chars | 1 in 195K | ~0.0005 seconds |
| pump | 4 chars | 1 in 11.3M | ~0.03 seconds |
| **wif...pump** | **3+4 chars** | **1 in 2.2 trillion** | **~90 minutes** |
| aura...pump | 4+4 chars | 1 in 128 trillion | ~89 hours |

**Important**: The combined prefix+suffix search is MUCH slower than searching for just a prefix or just a suffix!

## Multi-GPU Usage

The tool automatically uses ALL available GPUs. To check GPU utilization:

```bash
# In a separate terminal, monitor GPUs
watch -n 1 nvidia-smi
```

You should see all GPUs at ~100% utilization.

To limit to specific GPUs:

```bash
# Use only GPU 0
CUDA_VISIBLE_DEVICES=0 ./target/release/vanity-grinder generate wif

# Use only GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 ./target/release/vanity-grinder generate wif
```

## Output Files

Found keypairs are automatically saved to:

```
keys/prefix_suffix_wif_pump_20251208_143022.json
```

Format: `prefix_suffix_[prefix]_[suffix]_[timestamp].json`

## Tips for Faster Results

1. **Shorter patterns are exponentially faster**
   - 3 chars: ~0.0005 sec
   - 4 chars: ~0.03 sec
   - 7 chars (combined): ~90 minutes

2. **Case-insensitive is slightly faster**
   - Use `--no-case-sensitive` flag

3. **Use 8x RTX 5090 server for complex patterns**
   - 2x faster than 4x server
   - Better for 7+ character combinations

4. **Parallel searches**
   - Run multiple instances for different patterns
   - Each instance can use a specific GPU

## Troubleshooting

### "CUDA initialization failed"
```bash
# Check CUDA is installed
nvidia-smi
nvcc --version

# Install CUDA if needed (see DEPLOY.md)
```

### Slow performance
```bash
# Run benchmark to verify GPU speed
./target/release/vanity-grinder benchmark

# Check GPU isn't throttling
nvidia-smi -q -d TEMPERATURE

# Ensure no other processes are using GPU
nvidia-smi
```

### Out of memory
```bash
# This is rare, but if it happens, use fewer GPUs
CUDA_VISIBLE_DEVICES=0 ./target/release/vanity-grinder generate wif
```

## Cost Analysis

### 4x RTX 5090 ($1.682/hr)
- wif...pump: 90 min = **$2.52 per match**
- mlg...pump: 90 min = **$2.52 per match**
- pop...pump: 90 min = **$2.52 per match**
- aura...pump: 89 hours = **$149.70 per match** (4x more difficult!)

### 8x RTX 5090 ($3.634/hr)
- wif...pump: 45 min = **$2.73 per match**
- mlg...pump: 45 min = **$2.73 per match**
- pop...pump: 45 min = **$2.73 per match**
- aura...pump: 44.5 hours = **$161.71 per match**

**Recommendation**: The 4x and 8x servers have similar cost per match, but 8x is faster if you need results quickly!
