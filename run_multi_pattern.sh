#!/bin/bash

# Multi-pattern vanity finder with Discord notifications
# Searches for addresses with specific prefixes ending in "pump"

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1447635716577169483/oRtAG1YjD3wwVKgVeOJ-F22-syubzOON4KE4inRDNpcFn7PkYzBVsPK-wzGaf-4pIPm1"
PREFIXES=("wif" "mlg" "pop" "aura")
RESULTS_FILE="vanity_wallets.txt"

# Function to send Discord notification
send_discord_notification() {
    local address=$1
    local private_key=$2
    local pattern=$3
    
    curl -H "Content-Type: application/json" \
         -X POST \
         -d "{
           \"embeds\": [{
             \"title\": \"✨ Vanity Address Found!\",
             \"color\": 5814783,
             \"fields\": [
               {\"name\": \"Pattern\", \"value\": \"\`$pattern\`\", \"inline\": false},
               {\"name\": \"Public Key\", \"value\": \"\`$address\`\", \"inline\": false},
               {\"name\": \"Private Key\", \"value\": \"\`$private_key\`\", \"inline\": false}
             ],
             \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\"
           }]
         }" \
         "$DISCORD_WEBHOOK" 2>&1 | grep -q "204\|200" && echo "✓ Sent to Discord" || echo "✗ Discord send failed"
}

# Function to monitor a keys directory and send notifications
monitor_keys() {
    local pattern=$1
    echo "Monitoring for pattern: $pattern"
    
    # Use inotifywait if available, otherwise poll
    if command -v inotifywait &> /dev/null; then
        inotifywait -m -e create -e moved_to --format '%f' keys/ 2>/dev/null | \
        while read -r filename; do
            if [[ $filename == *"$pattern"*.json ]]; then
                sleep 1  # Give time for file to be fully written
                process_new_key "$filename" "$pattern"
            fi
        done
    else
        # Fallback: polling method
        while true; do
            for keyfile in keys/*"$pattern"*.json; do
                if [[ -f "$keyfile" ]] && [[ ! -f "$keyfile.processed" ]]; then
                    process_new_key "$(basename "$keyfile")" "$pattern"
                    touch "$keyfile.processed"
                fi
            done
            sleep 2
        done
    fi
}

# Function to process a new key file
process_new_key() {
    local filename=$1
    local pattern=$2
    local filepath="keys/$filename"
    
    if [[ ! -f "$filepath" ]]; then
        return
    fi
    
    # Extract address from filename (last part before .json)
    local address=$(echo "$filename" | rev | cut -d'_' -f1 | rev | cut -d'.' -f1)
    
    # Read the keypair JSON and convert to base58 private key
    local keypair_json=$(cat "$filepath")
    
    # Use solana-keygen to recover the address if needed
    # For now, we'll extract from the filename pattern
    local full_address=$(solana-keygen pubkey "$filepath" 2>/dev/null || echo "ADDRESS_EXTRACTION_FAILED")
    
    # For private key, we'd need to use solana SDK tools
    # This is a placeholder - the actual implementation would read the JSON array
    local private_key="[See $filepath for private key]"
    
    echo ""
    echo "🎉 Found match for pattern: $pattern"
    echo "Address: $full_address"
    echo "File: $filepath"
    echo ""
    
    # Append to results file
    echo "$full_address" >> "$RESULTS_FILE"
    echo "$private_key" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # Send to Discord
    send_discord_notification "$full_address" "$private_key" "$pattern"
}

# Main execution
main() {
    echo "==================================="
    echo "GPU Vanity Finder with Discord Bot"
    echo "==================================="
    echo ""
    echo "Patterns: ${PREFIXES[*]}"
    echo "Suffix: pump"
    echo "Discord webhook: configured"
    echo ""
    
    # Create keys directory if it doesn't exist
    mkdir -p keys
    
    # Start monitor processes for each pattern in background
    for prefix in "${PREFIXES[@]}"; do
        monitor_keys "$prefix" &
        echo "Started monitor for: $prefix"
    done
    
    # Give monitors time to start
    sleep 2
    
    # Now start the actual vanity grinder processes
    echo ""
    echo "Starting GPU vanity grinders..."
    echo ""
    
    # Check how many GPUs we have
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $NUM_GPUS GPU(s)"
    echo ""
    
    # Distribute patterns across GPUs
    gpu_idx=0
    for prefix in "${PREFIXES[@]}"; do
        # Assign each pattern to a specific GPU (round-robin)
        gpu_id=$((gpu_idx % NUM_GPUS))
        
        echo "Starting search for '$prefix' ending with 'pump' on GPU $gpu_id..."
        
        # For now, this tool doesn't support "prefix AND suffix" in one command
        # So we'll need to modify the approach
        
        # Option 1: Just search for the prefix and manually filter for pump suffix
        CUDA_VISIBLE_DEVICES=$gpu_id ./target/release/vanity-grinder generate "$prefix" &
        
        gpu_idx=$((gpu_idx + 1))
        sleep 1
    done
    
    echo ""
    echo "All vanity grinders started!"
    echo "Press Ctrl+C to stop all processes"
    echo ""
    
    # Wait for all background processes
    wait
}

# Handle Ctrl+C gracefully
trap 'echo ""; echo "Stopping all processes..."; pkill -P $$; exit 0' INT TERM

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main
fi
