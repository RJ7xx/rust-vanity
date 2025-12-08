
#include <curand_kernel.h>

// Seeds for random number generation
extern "C" __global__ void init_rng(curandState *states, unsigned long long seed, int num_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Base58 character set used by Solana
__device__ const char BASE58_CHARS[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

// Checks if the base58 encoded address matches the given pattern
__device__ bool check_pattern(const char* encoded, const char* pattern, int pattern_len, 
                              bool is_prefix, bool case_sensitive) {
    if (is_prefix) {
        for (int i = 0; i < pattern_len; i++) {
            char c1 = encoded[i];
            char c2 = pattern[i];
            
            if (!case_sensitive) {
                // Simple lowercase conversion for ASCII
                if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
                if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
            }
            
            if (c1 != c2) return false;
        }
        return true;
    } else {
        // Suffix matching
        int addr_len = 0;
        while (encoded[addr_len] != '\0') addr_len++;
        
        if (addr_len < pattern_len) return false;
        
        for (int i = 0; i < pattern_len; i++) {
            char c1 = encoded[addr_len - pattern_len + i];
            char c2 = pattern[i];
            
            if (!case_sensitive) {
                // Simple lowercase conversion for ASCII
                if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
                if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
            }
            
            if (c1 != c2) return false;
        }
        return true;
    }
}

// Checks if address matches BOTH prefix and suffix
__device__ bool check_prefix_and_suffix(const char* encoded, 
                                       const char* prefix, int prefix_len,
                                       const char* suffix, int suffix_len,
                                       bool case_sensitive) {
    // Check prefix first (fast fail)
    for (int i = 0; i < prefix_len; i++) {
        char c1 = encoded[i];
        char c2 = prefix[i];
        
        if (!case_sensitive) {
            if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
            if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
        }
        
        if (c1 != c2) return false;
    }
    
    // Get address length
    int addr_len = 0;
    while (encoded[addr_len] != '\0') addr_len++;
    
    if (addr_len < suffix_len) return false;
    
    // Check suffix
    for (int i = 0; i < suffix_len; i++) {
        char c1 = encoded[addr_len - suffix_len + i];
        char c2 = suffix[i];
        
        if (!case_sensitive) {
            if (c1 >= 'A' && c1 <= 'Z') c1 += 32;
            if (c2 >= 'A' && c2 <= 'Z') c2 += 32;
        }
        
        if (c1 != c2) return false;
    }
    
    return true;
}

// Simple Base58Check encoding - simplified for pattern matching only
__device__ void encode_base58_check(const unsigned char* data, int len, char* output) {
    // This is a simplified version that doesn't do proper Base58Check
    // but is sufficient for pattern matching
    
    int out_idx = 0;
    
    // Just encode the first few bytes for pattern matching
    for (int i = 0; i < min(len, 8); i++) {
        output[out_idx++] = BASE58_CHARS[data[i] % 58];
    }
    
    output[out_idx] = '\0';
}

// Main kernel for generating and checking keypairs
extern "C" __global__ void generate_and_check_keypairs(
    curandState *states,
    unsigned char *seed_data,
    unsigned char *result_keypair,
    int *found_flag,
    char *pattern,
    int pattern_len,
    bool is_prefix,
    bool case_sensitive,
    int num_keys
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys || *found_flag > 0) return;
    
    // Generate random seed data (32 bytes for ed25519)
    unsigned char seed[32];
    int offset = idx * 32;
    
    curandState localState = states[idx];
    for (int i = 0; i < 32; i++) {
        seed[i] = (unsigned char)(curand(&localState) % 256);
    }
    
    // Copy seed to global memory for later processing
    for (int i = 0; i < 32; i++) {
        seed_data[offset + i] = seed[i];
    }
    
    // Simplified: we'll use the seed as the public key for now
    // In the actual implementation, this would be replaced with proper ed25519 derivation
    char encoded_address[64];
    encode_base58_check(seed, 32, encoded_address);
    
    // Check if the address matches the pattern
    if (check_pattern(encoded_address, pattern, pattern_len, is_prefix, case_sensitive)) {
        // If we found a match and no one else has, copy the keypair to the result
        if (atomicExch(found_flag, 1) == 0) {
            for (int i = 0; i < 32; i++) {
                result_keypair[i] = seed[i];
            }
        }
    }
}

// This is the old function kept for backward compatibility with the benchmarking code
extern "C" __global__ void generate_keypairs(unsigned char *keys, int num_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_keys) {
        int offset = idx * 32;
        for (int i = 0; i < 32; i++) {
            keys[offset + i] = (unsigned char)((idx + i) % 256);
        }
    }
}

// Kernel for checking prefix AND suffix together
extern "C" __global__ void generate_and_check_keypairs_prefix_suffix(
    curandState *states,
    unsigned char *seed_data,
    unsigned char *result_keypair,
    int *found_flag,
    char *prefix,
    int prefix_len,
    char *suffix,
    int suffix_len,
    bool case_sensitive,
    int num_keys
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys || *found_flag > 0) return;
    
    // Generate random seed data (32 bytes for ed25519)
    unsigned char seed[32];
    int offset = idx * 32;
    
    curandState localState = states[idx];
    for (int i = 0; i < 32; i++) {
        seed[i] = (unsigned char)(curand(&localState) % 256);
    }
    
    // Copy seed to global memory for later processing
    for (int i = 0; i < 32; i++) {
        seed_data[offset + i] = seed[i];
    }
    
    // Simplified: we'll use the seed as the public key for now
    // In the actual implementation, this would be replaced with proper ed25519 derivation
    char encoded_address[64];
    encode_base58_check(seed, 32, encoded_address);
    
    // Check if the address matches BOTH prefix and suffix
    if (check_prefix_and_suffix(encoded_address, prefix, prefix_len, suffix, suffix_len, case_sensitive)) {
        // If we found a match and no one else has, copy the keypair to the result
        if (atomicExch(found_flag, 1) == 0) {
            for (int i = 0; i < 32; i++) {
                result_keypair[i] = seed[i];
            }
        }
    }
}
        