#!/usr/bin/env bash
# filepath: /home/hector/github/data-mining-hramirez-zubi/automate-data-source/run.sh

# Exit on error
set -e

# Activate virtual environment
source .venv/bin/activate

# Function to show usage information
show_usage() {
  echo "Usage: ./run.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --help             Show this help message"
  echo "  --start DATE       Start date (YYYY-MM-DD)"
  echo "  --end DATE         End date (YYYY-MM-DD)" 
  echo "  --mag NUMBER       Minimum magnitude"
  echo "  --limit NUMBER     Maximum number of results"
  echo "  --lat NUMBER       Latitude for center of search"
  echo "  --lon NUMBER       Longitude for center of search"
  echo "  --radius NUMBER    Search radius in kilometers"
  echo "  --loc LOCATION     Predefined location (sf, la, nyc, tokyo, mexico_city)"
  echo ""
  echo "Examples:"
  echo "  ./run.sh                                  # Run with defaults"
  echo "  ./run.sh --start 2023-01-01 --end 2023-12-31  # Custom date range"
  echo "  ./run.sh --mag 5.0 --limit 100           # Magnitude ≥ 5.0, max 100 results"
  echo "  ./run.sh --loc sf                         # Earthquakes near San Francisco"
  echo "  ./run.sh --lat 37.7749 --lon -122.4194 --radius 100  # Custom location"
}

# Parse command line arguments
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      show_usage
      exit 0
      ;;
    --start|--end|--mag|--limit|--lat|--lon|--radius|--loc|--output-dir)
      # For options that take a value
      if [[ -z "$2" || "$2" == --* ]]; then
        echo "Error: Missing value for parameter $1"
        exit 1
      fi
      
      # Map script arguments to python script arguments
      case $1 in
        --start) ARGS+=("-s" "$2") ;;
        --end) ARGS+=("-e" "$2") ;;
        --mag) ARGS+=("-m" "$2") ;;
        --limit) ARGS+=("-l" "$2") ;;
        --loc) ARGS+=("--location" "$2") ;;
        *) ARGS+=("$1" "$2") ;;
      esac
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

echo "Running data-gather.py with parameters: ${ARGS[*]}"
python3 data-gather.py "${ARGS[@]}"

echo "Done! Check the data directory for results."