#!/bin/bash

# Simple GPU vanity runner - finds addresses starting with your patterns
# Note: The base tool finds prefixes OR suffixes, not both at once
# To find "wifXXXXpump" you'd need to modify the CUDA kernel

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1447635716577169483/oRtAG1YjD3wwVKgVeOJ-F22-syubzOON4KE4inRDNpcFn7PkYzBVsPK-wzGaf-4pIPm1"

echo "================================"
echo "GPU Vanity Address Finder"
echo "================================"
echo ""

# Build the project if needed
if [[ ! -f target/release/vanity-grinder ]]; then
    echo "Building project..."
    cargo build --release || { echo "Build failed!"; exit 1; }
fi

# Run benchmark first
echo "Running GPU benchmark..."
./target/release/vanity-grinder benchmark

echo ""
echo "Choose your search mode:"
echo "1) Search for prefixes only (wif, mlg, pop, aura)"
echo "2) Search for suffix only (pump)"
echo "3) Custom pattern"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Searching for prefixes: wif, mlg, pop, aura"
        echo "Each will run on a separate GPU if available"
        echo ""
        
        # Run each pattern on a different GPU
        CUDA_VISIBLE_DEVICES=0 ./target/release/vanity-grinder generate wif &
        sleep 2
        CUDA_VISIBLE_DEVICES=1 ./target/release/vanity-grinder generate mlg &
        sleep 2
        CUDA_VISIBLE_DEVICES=2 ./target/release/vanity-grinder generate pop &
        sleep 2
        CUDA_VISIBLE_DEVICES=3 ./target/release/vanity-grinder generate aura &
        
        echo "All searches started. Press Ctrl+C to stop."
        wait
        ;;
    2)
        echo ""
        ./target/release/vanity-grinder generate pump --suffix
        ;;
    3)
        read -p "Enter pattern: " pattern
        read -p "Is this a suffix? [y/N]: " is_suffix
        
        if [[ $is_suffix == "y" || $is_suffix == "Y" ]]; then
            ./target/release/vanity-grinder generate "$pattern" --suffix
        else
            ./target/release/vanity-grinder generate "$pattern"
        fi
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
