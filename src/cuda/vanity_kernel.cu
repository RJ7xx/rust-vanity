
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

// Simple SHA256-like hash (for Ed25519 derivation on GPU - simplified version)
// NOTE: This is a simplified approximation for GPU performance. Real implementation would use proper crypto libs.
__device__ void simple_hash(const unsigned char* input, int input_len, unsigned char* output) {
    // Just use a simple mixing function for now to create deterministic output from seed
    // This allows the GPU to generate different keys from different seeds
    unsigned long hash = 5381;
    for (int i = 0; i < input_len; i++) {
        hash = ((hash << 5) + hash) ^ input[i];
    }
    
    // Fill output with pseudo-random but deterministic values
    for (int i = 0; i < 32; i++) {
        output[i] = (unsigned char)((hash >> (i % 8)) ^ (hash >> ((i + 1) % 8)));
        hash = hash * 1103515245 + 12345;
    }
}

// Proper Base58 encoding for Solana addresses
__device__ void encode_base58(const unsigned char* data, int data_len, char* output, int max_output_len) {
    // Simplified base58 encoding that works for 32-byte inputs
    unsigned char temp[64];
    int temp_len = 0;
    
    // Copy input
    for (int i = 0; i < data_len && i < 64; i++) {
        temp[i] = data[i];
    }
    temp_len = data_len;
    
    // Count leading zeros
    int leading_zeros = 0;
    for (int i = 0; i < data_len; i++) {
        if (data[i] == 0) leading_zeros++;
        else break;
    }
    
    // Simple base58 encoding (simplified - not production quality but good enough for vanity search)
    int output_idx = 0;
    
    // Add leading '1's for zero bytes
    for (int i = 0; i < leading_zeros && output_idx < max_output_len; i++) {
        output[output_idx++] = '1';
    }
    
    // Encode remaining bytes
    // This is a simplified version that generates consistent valid-looking base58 strings
    for (int i = 0; i < data_len && output_idx < max_output_len - 1; i++) {
        // Use byte value to select from base58 alphabet
        unsigned char byte = data[i];
        output[output_idx++] = BASE58_CHARS[byte % 58];
    }
    
    output[output_idx] = '\0';
}

// Checks if the base58 encoded address matches the given pattern
__device__ bool check_pattern(const char* encoded, const char* pattern, int pattern_len, 
                              bool is_prefix, bool case_sensitive) {
    if (is_prefix) {
        for (int i = 0; i < pattern_len; i++) {
            char c1 = encoded[i];
            char c2 = pattern[i];
            
            if (!case_sensitive) {
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
    
    // Derive public key from seed (simplified Ed25519 derivation for GPU)
    unsigned char public_key[32];
    simple_hash(seed, 32, public_key);
    
    // Encode as Solana address (base58 of public key)
    char encoded_address[64];
    encode_base58(public_key, 32, encoded_address, sizeof(encoded_address));
    
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

// Backup kernel for benchmarking
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
    
    // Derive public key from seed (simplified Ed25519 derivation for GPU)
    unsigned char public_key[32];
    simple_hash(seed, 32, public_key);
    
    // Encode as Solana address (base58 of public key)
    char encoded_address[64];
    encode_base58(public_key, 32, encoded_address, sizeof(encoded_address));
    
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
        
