#!/bin/bash
# DEPRECATED: Use init/gpu/start-gt5.sh instead
echo "This script has moved to init/gpu/start-gt5.sh"
echo "Running new script..."
exec bash "$(dirname "$0")/gpu/start-gt5.sh" "$@"
